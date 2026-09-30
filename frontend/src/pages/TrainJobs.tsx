import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Empty,
  Popconfirm,
  Progress,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  PlusOutlined,
  ReloadOutlined,
  StopOutlined,
  RedoOutlined,
  EyeOutlined,
  CloudServerOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { TrainJob } from '../types'
import StatusTag, { isActiveStatus } from '../components/StatusTag'
import { formatMetric, formatTime } from '../utils'

const { Text, Paragraph } = Typography

function bestHeadline(job: TrainJob): string {
  const nested = (job.best?.best ?? {}) as Record<string, number>
  const key = ['mAP50-95', 'mAP50', 'top1'].find((k) => nested[k] != null)
  if (!key) return '-'
  return `${key} ${formatMetric(nested[key])}`
}

/** 训练任务列表（M5-07 的列表部分）。 */
export default function TrainJobs() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState<TrainJob[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const timer = useRef<number>()

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const res = await api.trainJobs()
      setJobs(res.jobs)
      setError('')
    } catch (err: any) {
      setError(err.message ?? '读取训练任务失败')
    } finally {
      if (!silent) setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  // 有任务在跑时静默轮询，结束后自动停止（实时细节在详情页看，这里只要列表状态新）
  useEffect(() => {
    const hasActive = jobs.some((j) => isActiveStatus(j.status))
    if (!hasActive) return
    timer.current = window.setInterval(() => load(true), 3000)
    return () => window.clearInterval(timer.current)
  }, [jobs, load])

  const doStop = async (id: string) => {
    try {
      await api.stopTrain(id)
      message.success('已发送停止指令')
      load(true)
    } catch (err: any) {
      message.error(err.message ?? '停止失败')
    }
  }

  const doResume = async (id: string) => {
    try {
      await api.resumeTrain(id)
      message.success('已从 last.pt 续训')
      load(true)
    } catch (err: any) {
      message.error(err.message ?? '续训失败')
    }
  }

  const doRegister = async (id: string) => {
    try {
      await api.registerModel(id)
      message.success('已注册到模型库')
    } catch (err: any) {
      message.error(err.message ?? '注册失败')
    }
  }

  return (
    <div className="page">
      <Card
        title="训练任务"
        size="small"
        extra={
          <Space>
            <Button size="small" icon={<ReloadOutlined />} onClick={() => load()} loading={loading}>
              刷新
            </Button>
            <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => navigate('/train/new')}>
              新建训练
            </Button>
          </Space>
        }
      >
        <Paragraph type="secondary">
          训练跑在独立子进程里，崩溃不影响本服务；指标实时读自训练目录的
          <Text code>results.csv</Text>，日志同时落盘到 <Text code>train.log</Text>。
          点任务名进入详情页看实时曲线、资源与日志。
        </Paragraph>

        {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}

        {jobs.length === 0 && !loading ? (
          <Empty description="还没有训练任务，点击右上角「新建训练」开始" />
        ) : (
          <Table
            size="small"
            rowKey="id"
            loading={loading}
            dataSource={jobs}
            pagination={false}
            scroll={{ x: 'max-content' }}
            columns={[
              {
                title: '任务',
                dataIndex: 'id',
                render: (id: string, r) => (
                  <Space direction="vertical" size={0}>
                    <a onClick={() => navigate(`/train/${encodeURIComponent(id)}`)}>
                      <Text strong>{id}</Text>
                    </a>
                    <Text type="secondary" className="mono" style={{ fontSize: 11 }}>
                      {r.run_dir}
                    </Text>
                  </Space>
                ),
              },
              {
                title: '状态',
                dataIndex: 'status',
                width: 110,
                render: (s: string, r) => <StatusTag status={s} label={r.status_label} />,
              },
              {
                title: '进度',
                width: 190,
                render: (_, r) => (
                  <div>
                    <Progress
                      percent={Math.round((r.progress ?? 0) * 100)}
                      size="small"
                      status={r.status === 'failed' ? 'exception' : undefined}
                    />
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      epoch {r.current_epoch}/{r.epochs_total || '-'}（{r.metrics_rows} 轮已记录）
                    </Text>
                  </div>
                ),
              },
              {
                title: '最优指标',
                width: 150,
                render: (_, r) => (
                  <Space direction="vertical" size={0}>
                    <Text>{bestHeadline(r)}</Text>
                    {r.best?.best_epoch != null && (
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        @ epoch {r.best.best_epoch}
                      </Text>
                    )}
                  </Space>
                ),
              },
              {
                title: '创建时间',
                dataIndex: 'created_at',
                width: 165,
                render: (v: string) => <span className="nowrap">{v || '-'}</span>,
              },
              {
                title: '操作',
                width: 230,
                render: (_, r) => (
                  <Space size={4} wrap>
                    <Button
                      size="small"
                      type="link"
                      icon={<EyeOutlined />}
                      onClick={() => navigate(`/train/${encodeURIComponent(r.id)}`)}
                    >
                      详情
                    </Button>
                    {isActiveStatus(r.status) ? (
                      <Popconfirm title="确定停止该训练？" onConfirm={() => doStop(r.id)}>
                        <Button size="small" type="link" danger icon={<StopOutlined />}>
                          停止
                        </Button>
                      </Popconfirm>
                    ) : (
                      <Tooltip title="从 weights/last.pt 继续训练">
                        <Button
                          size="small"
                          type="link"
                          icon={<RedoOutlined />}
                          onClick={() => doResume(r.id)}
                        >
                          续训
                        </Button>
                      </Tooltip>
                    )}
                    {r.status === 'finished' && (
                      <Button
                        size="small"
                        type="link"
                        icon={<CloudServerOutlined />}
                        onClick={() => doRegister(r.id)}
                      >
                        注册
                      </Button>
                    )}
                  </Space>
                ),
              },
            ]}
          />
        )}
      </Card>

      <Card size="small" style={{ marginTop: 16 }}>
        <Space wrap>
          <Tag>任务 id 即训练目录名</Tag>
          <Tag>停止 = 终止整个进程树</Tag>
          <Tag>续训 = 复用 last.pt，指标按 epoch 去重继续累积</Tag>
        </Space>
      </Card>
    </div>
  )
}
