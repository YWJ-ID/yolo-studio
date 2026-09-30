import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Popconfirm,
  Progress,
  Row,
  Space,
  Spin,
  Statistic,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  ArrowLeftOutlined,
  CloudServerOutlined,
  RedoOutlined,
  ReloadOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type {
  ArtifactsResponse,
  LogLine,
  MetricsSeries,
  ResourceSample,
  TrainEvent,
  TrainJob,
} from '../types'
import StatusTag, { isActiveStatus } from '../components/StatusTag'
import MetricsChart from '../components/MetricsChart'
import ResourcePanel from '../components/ResourcePanel'
import LogViewer from '../components/LogViewer'
import ArtifactGallery from '../components/ArtifactGallery'
import { useJobSocket } from '../hooks/useJobSocket'

const { Text, Paragraph } = Typography

const HEADLINE_KEYS = ['mAP50-95', 'mAP50', 'precision', 'recall', 'f1', 'top1', 'top5', 'train_loss']

function mergeLogs(prev: LogLine[], incoming: LogLine[]): LogLine[] {
  if (!incoming?.length) return prev
  const last = prev.length ? prev[prev.length - 1].seq : -1
  const added = incoming.filter((l) => l.seq > last)
  return added.length ? [...prev, ...added] : prev
}

function mergeMetrics(prev: MetricsSeries | null, ev: TrainEvent): MetricsSeries {
  const base: MetricsSeries =
    prev ?? { columns: [], epochs: 0, last_epoch: null, rows: [], best: {} }
  const seen = new Set(base.rows.map((r) => r.epoch))
  const added = (ev.rows ?? []).filter((r) => !seen.has(r.epoch))
  const rows = added.length ? [...base.rows, ...added].sort((a, b) => a.epoch - b.epoch) : base.rows
  return {
    ...base,
    rows,
    epochs: rows.length,
    last_epoch: rows.length ? rows[rows.length - 1].epoch : null,
    best: ev.best ?? base.best,
  }
}

/** 训练详情（M5-05）：实时曲线 + 资源 + 日志，全部走 WebSocket 增量推送。 */
export default function TrainDetail() {
  const { jobId = '' } = useParams()
  const navigate = useNavigate()

  const [job, setJob] = useState<TrainJob | null>(null)
  const [metrics, setMetrics] = useState<MetricsSeries | null>(null)
  const [resources, setResources] = useState<ResourceSample | null>(null)
  const [monitor, setMonitor] = useState<TrainEvent['monitor'] | null>(null)
  const [logs, setLogs] = useState<LogLine[]>([])
  const [artifacts, setArtifacts] = useState<ArtifactsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const finishedRef = useRef(false)

  const refreshArtifacts = useCallback(async () => {
    try {
      const res = await api.trainArtifacts(jobId)
      setArtifacts(res)
    } catch {
      // 过程图在训练早期可能还不存在，忽略
    }
  }, [jobId])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.trainJob(jobId)
      setJob(res.job)
      setMetrics(res.metrics)
      setResources(res.resources)
      setLogs(res.logs?.lines ?? [])
      setError('')
      finishedRef.current = ['finished', 'failed', 'stopped', 'interrupted'].includes(res.job.status)
    } catch (err: any) {
      setError(err.message ?? '读取训练任务失败')
    } finally {
      setLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    load()
    refreshArtifacts()
  }, [load, refreshArtifacts])

  const onEvent = useCallback(
    (ev: TrainEvent) => {
      switch (ev.type) {
        case 'snapshot':
          if (ev.job) setJob(ev.job)
          if (ev.metrics) setMetrics(ev.metrics)
          if (ev.resources) setResources(ev.resources)
          if (ev.monitor) setMonitor(ev.monitor)
          if (ev.logs?.lines) setLogs(ev.logs.lines)
          break
        case 'metrics': {
          setMetrics((prev) => mergeMetrics(prev, ev))
          const prog = typeof ev.progress === 'object' ? ev.progress : undefined
          setJob((prev) =>
            prev
              ? {
                  ...prev,
                  progress: prog?.percent ?? prev.progress,
                  current_epoch: prog?.epoch ?? prev.current_epoch,
                  metrics_rows: (prev.metrics_rows ?? 0) + (ev.rows?.length ?? 0),
                  best: ev.best ?? prev.best,
                }
              : prev,
          )
          break
        }
        case 'log':
          setLogs((prev) => mergeLogs(prev, ev.lines ?? []))
          break
        case 'resources':
          if (ev.sample) setResources(ev.sample)
          break
        case 'status':
          setJob((prev) =>
            prev
              ? {
                  ...prev,
                  status: ev.status ?? prev.status,
                  status_label: ev.status_label ?? prev.status_label,
                  message: ev.message ?? prev.message,
                  error: ev.error ?? prev.error,
                  progress: typeof ev.progress === 'number' ? ev.progress : prev.progress,
                  current_epoch: ev.current_epoch ?? prev.current_epoch,
                }
              : prev,
          )
          break
        case 'finished':
          setJob((prev) =>
            prev
              ? {
                  ...prev,
                  status: ev.status ?? prev.status,
                  returncode: ev.returncode ?? prev.returncode,
                }
              : prev,
          )
          if (!finishedRef.current) {
            finishedRef.current = true
            // finished 之前后端已先完成「注册模型 / 自动评估」联动，这里只需刷新产物
            refreshArtifacts()
            api.trainJob(jobId).then((r) => setJob(r.job)).catch(() => undefined)
          }
          break
      }
    },
    [jobId, refreshArtifacts],
  )

  const { connected } = useJobSocket<TrainEvent>(jobId ? `/api/train/jobs/${jobId}/ws` : null, onEvent)

  const doStop = async () => {
    try {
      await api.stopTrain(jobId)
      message.success('已发送停止指令')
    } catch (err: any) {
      message.error(err.message ?? '停止失败')
    }
  }

  const doResume = async () => {
    try {
      await api.resumeTrain(jobId)
      message.success('已从 last.pt 续训')
      load()
    } catch (err: any) {
      message.error(err.message ?? '续训失败')
    }
  }

  const doRegister = async () => {
    try {
      await api.registerModel(jobId)
      message.success('已注册到模型库')
    } catch (err: any) {
      message.error(err.message ?? '注册失败')
    }
  }

  const spec = job?.spec ?? {}
  const nestedBest = (job?.best?.best ?? {}) as Record<string, number>

  const bestStats = useMemo(
    () =>
      HEADLINE_KEYS.filter((k) => nestedBest[k] != null).map((k) => ({
        key: k,
        value: nestedBest[k],
      })),
    [nestedBest],
  )

  if (loading) return <Spin style={{ display: 'block', marginTop: 80 }} />

  if (error && !job) {
    return (
      <div className="page">
        <Alert
          type="error"
          showIcon
          message="读取训练任务失败"
          description={error}
          action={
            <Button size="small" onClick={() => navigate('/train')}>
              返回列表
            </Button>
          }
        />
      </div>
    )
  }

  if (!job) return null

  const active = isActiveStatus(job.status)

  return (
    <div className="page">
      <Card
        size="small"
        title={
          <Space wrap>
            <Text strong>{job.id}</Text>
            <StatusTag status={job.status} label={job.status_label} />
            {connected ? (
              <Tag color="green">实时已连接</Tag>
            ) : active ? (
              <Tag color="orange">实时未连接</Tag>
            ) : (
              <Tag>已结束</Tag>
            )}
          </Space>
        }
        extra={
          <Space>
            <Button size="small" icon={<ArrowLeftOutlined />} onClick={() => navigate('/train')}>
              返回列表
            </Button>
            <Button size="small" icon={<ReloadOutlined />} onClick={load}>
              刷新
            </Button>
            {active ? (
              <Popconfirm title="确定停止该训练？" onConfirm={doStop}>
                <Button size="small" danger icon={<StopOutlined />}>
                  停止
                </Button>
              </Popconfirm>
            ) : (
              <Button size="small" icon={<RedoOutlined />} onClick={doResume}>
                续训
              </Button>
            )}
            {job.status === 'finished' && (
              <Button size="small" type="primary" icon={<CloudServerOutlined />} onClick={doRegister}>
                注册模型
              </Button>
            )}
          </Space>
        }
      >
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} md={10}>
            <Progress
              percent={Math.round((job.progress ?? 0) * 100)}
              status={job.status === 'failed' ? 'exception' : active ? 'active' : 'success'}
            />
            <Text type="secondary">
              epoch {job.current_epoch}/{job.epochs_total || '-'}　·　{job.metrics_rows} 轮已记录
            </Text>
          </Col>
          <Col xs={24} md={14}>
            <Space wrap size={16}>
              {bestStats.length ? (
                bestStats.map((s) => (
                  <Statistic
                    key={s.key}
                    title={s.key}
                    value={s.value}
                    precision={4}
                    valueStyle={{ fontSize: 18 }}
                  />
                ))
              ) : (
                <Text type="secondary">尚无指标（等待第一轮 results.csv）</Text>
              )}
              {job.best?.best_epoch != null && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  （最优 @ epoch {job.best.best_epoch}）
                </Text>
              )}
            </Space>
          </Col>
        </Row>

        {job.message && (
          <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
            {job.message}
          </Paragraph>
        )}
        {job.error && (
          <Alert type="error" showIcon style={{ marginTop: 8 }} message={job.error} />
        )}
        {job.status === 'finished' && (
          <Alert
            type="success"
            showIcon
            style={{ marginTop: 8 }}
            message="训练完成，已自动注册到模型库并触发评估"
            description="可到「模型库」查看评估指标、逐类指标与导出部署。"
          />
        )}
      </Card>

      <Card title="实时指标曲线" size="small" style={{ marginTop: 16 }}>
        <MetricsChart series={metrics} />
      </Card>

      <Card title="资源占用" size="small" style={{ marginTop: 16 }}>
        <ResourcePanel sample={resources} monitor={monitor} />
      </Card>

      <Card
        title="训练日志"
        size="small"
        style={{ marginTop: 16 }}
        extra={<Text type="secondary" style={{ fontSize: 12 }}>stdout + stderr，落盘到 train.log</Text>}
      >
        <LogViewer
          lines={logs}
          connected={connected}
          onClear={() => setLogs([])}
          emptyText="暂无日志（训练进程启动后开始输出）"
        />
      </Card>

      <Card
        title="过程图像"
        size="small"
        style={{ marginTop: 16 }}
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={refreshArtifacts}>
            刷新
          </Button>
        }
      >
        {artifacts ? <ArtifactGallery data={artifacts} /> : <Empty description="暂无过程图像" />}
      </Card>

      <Card title="训练参数与产物" size="small" style={{ marginTop: 16 }}>
        <Descriptions column={{ xs: 1, md: 2 }} size="small" bordered>
          <Descriptions.Item label="data.yaml">
            <Text copyable className="mono" style={{ fontSize: 11 }}>
              {String(spec.data_yaml ?? '-')}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="权重">{String(spec.weights ?? '-')}</Descriptions.Item>
          <Descriptions.Item label="任务">{String(spec.task ?? '-')}</Descriptions.Item>
          <Descriptions.Item label="设备">{String(spec.device ?? '-')}</Descriptions.Item>
          <Descriptions.Item label="epochs / imgsz / batch">
            {String(spec.epochs ?? '-')} / {String(spec.imgsz ?? '-')} / {String(spec.batch ?? '-')}
          </Descriptions.Item>
          <Descriptions.Item label="optimizer / lr0">
            {String(spec.optimizer ?? '-')} / {spec.lr0 != null ? String(spec.lr0) : '自动'}
          </Descriptions.Item>
          <Descriptions.Item label="seed / patience">
            {String(spec.seed ?? '-')} / {String(spec.patience ?? '-')}
          </Descriptions.Item>
          <Descriptions.Item label="训练目录">
            <Text copyable className="mono" style={{ fontSize: 11 }}>
              {job.run_dir}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="创建 / 开始 / 结束">
            {job.created_at || '-'} / {job.started_at || '-'} / {job.ended_at || '-'}
          </Descriptions.Item>
          <Descriptions.Item label="退出码 / 尝试次数">
            {job.returncode != null ? job.returncode : '-'} / {job.attempts}
          </Descriptions.Item>
        </Descriptions>
        {metrics?.columns?.length ? (
          <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
            results.csv 列（{metrics.columns.length}）：{metrics.columns.join(', ')}
          </Paragraph>
        ) : null}
      </Card>
    </div>
  )
}
