import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Empty,
  InputNumber,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  ArrowLeftOutlined,
  CloudDownloadOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons'
import { useNavigate, useParams } from 'react-router-dom'
import { api, deployDownloadUrl } from '../api/client'
import type {
  ArtifactsResponse,
  DeployFormat,
  DeployJob,
  DeployResult,
  DeployVerifyReport,
  EvalJob,
  EvalResult,
  ModelCard,
} from '../types'
import EvalResultView from '../components/EvalResultView'
import ArtifactGallery from '../components/ArtifactGallery'
import StatusTag, { isActiveStatus } from '../components/StatusTag'
import { formatBytes } from '../utils'

const { Text, Paragraph } = Typography

/** 模型详情（M5-08）：概览 + 评估 + 导出部署。 */
export default function ModelDetail() {
  const { modelId = '' } = useParams()
  const navigate = useNavigate()
  const [card, setCard] = useState<ModelCard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadCard = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.model(modelId)
      setCard(res.model)
      setError('')
    } catch (err: any) {
      setError(err.message ?? '读取模型失败')
    } finally {
      setLoading(false)
    }
  }, [modelId])

  useEffect(() => {
    loadCard()
  }, [loadCard])

  if (loading && !card) return <Spin style={{ display: 'block', marginTop: 80 }} />

  if (error && !card) {
    return (
      <div className="page">
        <Alert
          type="error"
          showIcon
          message="读取模型失败"
          description={error}
          action={
            <Button size="small" onClick={() => navigate('/models')}>
              返回模型库
            </Button>
          }
        />
      </div>
    )
  }

  if (!card) return null

  return (
    <div className="page">
      <Card
        size="small"
        title={
          <Space wrap>
            <Text strong>{card.name || card.model_id}</Text>
            <Tag color={card.task === 'classify' ? 'purple' : 'blue'}>
              {card.task === 'classify' ? '图像分类' : '目标检测'}
            </Tag>
            {card.training?.source === 'external' && (
              <Tag color="orange" title="由外部 .pt 权重导入，非本项目训练任务">
                外部导入
              </Tag>
            )}
            <Tag>{card.classes.length} 类</Tag>
            <Tag>{card.evals.length} 次评估</Tag>
            <Tag>{card.deploys.length} 次导出</Tag>
          </Space>
        }
        extra={
          <Space>
            <Button size="small" icon={<ReloadOutlined />} onClick={loadCard}>
              刷新
            </Button>
            <Button size="small" icon={<ArrowLeftOutlined />} onClick={() => navigate('/models')}>
              返回模型库
            </Button>
          </Space>
        }
      >
        <Descriptions column={{ xs: 1, md: 2 }} size="small">
          <Descriptions.Item label="模型 id">
            <Text copyable className="mono">
              {card.model_id}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="训练任务">
            {card.training?.source === 'external' ? (
              <Text type="secondary">外部导入（无训练任务）</Text>
            ) : (
              card.training?.job_id ?? '-'
            )}
          </Descriptions.Item>
          <Descriptions.Item label="best 权重">
            {card.weights?.best ? (
              <Text copyable className="mono" style={{ fontSize: 11 }}>
                {card.weights.best}
              </Text>
            ) : (
              <Text type="secondary">无</Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="last 权重">
            {card.weights?.last ? (
              <Text copyable className="mono" style={{ fontSize: 11 }}>
                {card.weights.last}
              </Text>
            ) : (
              <Text type="secondary">无</Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="数据集">{card.dataset?.name ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="创建 / 更新">
            {card.created_at || '-'} / {card.updated_at || '-'}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Tabs
        style={{ marginTop: 16 }}
        items={[
          { key: 'overview', label: '概览', children: <OverviewPanel card={card} /> },
          {
            key: 'eval',
            label: `评估（${card.evals.length}）`,
            children: <EvalPanel modelId={modelId} card={card} onChanged={loadCard} />,
          },
          {
            key: 'deploy',
            label: `导出部署（${card.deploys.length}）`,
            children: <DeployPanel modelId={modelId} card={card} onChanged={loadCard} />,
          },
        ]}
      />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 概览                                                                */
/* ------------------------------------------------------------------ */

function OverviewPanel({ card }: { card: ModelCard }) {
  const best = (card.training?.best ?? {}) as Record<string, any>
  const nested = (best.best ?? {}) as Record<string, number>
  const keys = ['mAP50-95', 'mAP50', 'precision', 'recall', 'top1', 'train_loss'].filter(
    (k) => nested[k] != null,
  )
  const dataset = card.dataset ?? {}

  return (
    <>
      <Card title="训练最优指标" size="small">
        {keys.length ? (
          <Space wrap size={20}>
            {keys.map((k) => (
              <Statistic key={k} title={k} value={nested[k]} precision={4} valueStyle={{ fontSize: 18 }} />
            ))}
            {best.best_epoch != null && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                （@ epoch {best.best_epoch}）
              </Text>
            )}
          </Space>
        ) : (
          <Empty
            description={
              card.training?.source === 'external'
                ? '外部导入模型没有训练指标；请到「评估」页发起一次评估'
                : '训练记录里没有可展示的最优指标'
            }
          />
        )}
      </Card>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="训练参数" size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="epochs / imgsz / batch">
                {card.training?.epochs ?? '-'} / {card.training?.imgsz ?? '-'} / {card.training?.batch ?? '-'}
              </Descriptions.Item>
              <Descriptions.Item label="device">{card.training?.device ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="optimizer / seed">
                {card.training?.optimizer ?? '-'} / {card.training?.seed ?? '-'}
              </Descriptions.Item>
              <Descriptions.Item label="初始权重">
                {card.training?.weights_source ?? '-'}
              </Descriptions.Item>
              <Descriptions.Item label="训练目录">
                <Text copyable className="mono" style={{ fontSize: 11 }}>
                  {card.training?.run_dir ?? '-'}
                </Text>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="数据集血缘" size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="数据集名">{dataset.name ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="data.yaml">
                <Text copyable className="mono" style={{ fontSize: 11 }}>
                  {dataset.data_yaml ?? '-'}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="各子集图像数">
                {dataset.images_exported ? (
                  Object.entries(dataset.images_exported as Record<string, number>).map(([k, v]) => (
                    <Tag key={k}>
                      {k}: {v}
                    </Tag>
                  ))
                ) : (
                  <Text type="secondary">未记录</Text>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="来源">
                {(dataset.sources as any[])?.length ? (
                  (dataset.sources as any[]).map((s, i) => (
                    <div key={i} className="mono" style={{ fontSize: 11 }}>
                      [{s.format}] {s.root}
                    </div>
                  ))
                ) : (
                  <Text type="secondary">未记录</Text>
                )}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>

      <Card title="类别" size="small" style={{ marginTop: 16 }}>
        <Space wrap>
          {card.classes.map((c, i) => (
            <Tag key={c}>
              {i}: {c}
            </Tag>
          ))}
        </Space>
      </Card>

      {card.notes.length > 0 && (
        <Card title="备注" size="small" style={{ marginTop: 16 }}>
          <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
            {card.notes.map((n, i) => (
              <li key={i}>
                <Text type="secondary">{n}</Text>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/* 评估                                                                */
/* ------------------------------------------------------------------ */

function EvalPanel({
  modelId,
  card,
  onChanged,
}: {
  modelId: string
  card: ModelCard
  onChanged: () => void
}) {
  const [split, setSplit] = useState('auto')
  const [starting, setStarting] = useState(false)
  const [job, setJob] = useState<EvalJob | null>(null)
  // 没有 data.yaml 就无法评估（需要真值）；外部导入时常见
  const canEval = Boolean((card.dataset ?? {}).data_yaml)

  const evalSplits = useMemo(
    () => Array.from(new Set(card.evals.map((e) => e.split).filter(Boolean))) as string[],
    [card],
  )
  const [viewSplit, setViewSplit] = useState<string | undefined>(undefined)
  const [result, setResult] = useState<EvalResult | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [evalArtifacts, setEvalArtifacts] = useState<ArtifactsResponse | null>(null)
  const [reportBusy, setReportBusy] = useState(false)

  useEffect(() => {
    if (viewSplit === undefined && evalSplits.length) setViewSplit(evalSplits[evalSplits.length - 1])
  }, [evalSplits, viewSplit])

  const selectedEntry = useMemo(() => {
    const list = viewSplit ? card.evals.filter((e) => e.split === viewSplit) : card.evals
    return list.length ? list[list.length - 1] : undefined
  }, [card, viewSplit])

  const loadResult = useCallback(async () => {
    if (!card.evals.length) {
      setResult(null)
      return
    }
    setResultLoading(true)
    try {
      const res = await api.modelResult(modelId, viewSplit)
      setResult(res.result)
      if (selectedEntry?.eval_id) {
        try {
          setEvalArtifacts(await api.evalArtifacts(selectedEntry.eval_id))
        } catch {
          setEvalArtifacts(null)
        }
      }
    } catch (err: any) {
      setResult(null)
      setEvalArtifacts(null)
    } finally {
      setResultLoading(false)
    }
  }, [card.evals.length, modelId, viewSplit, selectedEntry])

  useEffect(() => {
    loadResult()
  }, [loadResult])

  // 轮询正在进行的评估任务
  useEffect(() => {
    if (!job || !isActiveStatus(job.status)) return
    const t = window.setInterval(async () => {
      try {
        const res = await api.evalJob(job.id)
        setJob(res.job)
        if (!isActiveStatus(res.job.status)) {
          message.success('评估已结束')
          onChanged()
        }
      } catch {
        // 忽略单次轮询失败
      }
    }, 2000)
    return () => window.clearInterval(t)
  }, [job?.id, job?.status, onChanged])

  const startEval = async () => {
    setStarting(true)
    try {
      const res = await api.evalModel(modelId, { split })
      setJob(res.job)
      message.success('评估已启动')
    } catch (err: any) {
      message.error(err.message ?? '启动评估失败')
    } finally {
      setStarting(false)
    }
  }

  const genReport = async () => {
    if (!selectedEntry?.eval_id) return
    setReportBusy(true)
    try {
      const res = await api.evalReport(selectedEntry.eval_id, {
        title: `${card.name || modelId} · ${selectedEntry.split} 评估报告`,
        embed_images: true,
      })
      message.success(`报告已生成（${(res.size_bytes / 1024).toFixed(1)} KB）`)
      window.open(res.url, '_blank')
    } catch (err: any) {
      message.error(err.message ?? '生成报告失败')
    } finally {
      setReportBusy(false)
    }
  }

  return (
    <>
      <Card title="发起评估" size="small">
        {!canEval && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 12 }}
            message="该模型没有关联 data.yaml，无法评估"
            description="评估需要带真值标注的数据集。请在导入外部模型时填写 data.yaml（或重新导入一次补上），再发起评估。"
          />
        )}
        <Space wrap>
          <Select
            style={{ width: 150 }}
            value={split}
            onChange={setSplit}
            options={[
              { value: 'auto', label: 'auto（有 test 用 test）' },
              { value: 'val', label: 'val' },
              { value: 'test', label: 'test' },
              { value: 'train', label: 'train' },
            ]}
          />
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            loading={starting}
            disabled={!canEval}
            onClick={startEval}
          >
            开始评估
          </Button>
          <Text type="secondary">评估跑在独立子进程，结果会自动挂到本模型卡片上</Text>
        </Space>

        {job && (
          <Alert
            style={{ marginTop: 12 }}
            type={job.status === 'finished' ? 'success' : job.status === 'failed' ? 'error' : 'info'}
            showIcon
            message={
              <Space>
                <span>评估任务 {job.id}</span>
                <StatusTag status={job.status} label={job.status_label} />
              </Space>
            }
            description={job.error || job.message || undefined}
          />
        )}
      </Card>

      <Card
        title="评估结果"
        size="small"
        style={{ marginTop: 16 }}
        extra={
          <Space>
            {evalSplits.length > 0 && (
              <Select
                size="small"
                style={{ width: 130 }}
                value={viewSplit}
                onChange={setViewSplit}
                options={evalSplits.map((s) => ({ value: s, label: `划分 ${s}` }))}
              />
            )}
            <Button
              size="small"
              icon={<FileTextOutlined />}
              disabled={!selectedEntry?.eval_id}
              loading={reportBusy}
              onClick={genReport}
            >
              HTML 报告
            </Button>
          </Space>
        }
      >
        {!card.evals.length ? (
          <Empty description="还没有评估结果，点击上方「开始评估」" />
        ) : resultLoading ? (
          <Spin style={{ display: 'block', margin: '24px auto' }} />
        ) : result ? (
          <EvalResultView result={result} />
        ) : (
          <Empty description="读取评估结果失败" />
        )}
      </Card>

      {evalArtifacts && evalArtifacts.images.length > 0 && (
        <Card title="评估过程图像" size="small" style={{ marginTop: 16 }}>
          <ArtifactGallery data={evalArtifacts} />
        </Card>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/* 导出部署                                                            */
/* ------------------------------------------------------------------ */

function DeployPanel({
  modelId,
  card,
  onChanged,
}: {
  modelId: string
  card: ModelCard
  onChanged: () => void
}) {
  const [formats, setFormats] = useState<DeployFormat[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [imgsz, setImgsz] = useState<number>(Number(card.training?.imgsz) || 640)
  const [starting, setStarting] = useState(false)
  const [job, setJob] = useState<DeployJob | null>(null)
  const [result, setResult] = useState<DeployResult | null>(null)
  const [verify, setVerify] = useState<DeployVerifyReport | null>(null)
  const [verifying, setVerifying] = useState(false)

  useEffect(() => {
    api
      .deployFormats()
      .then((res) => {
        setFormats(res.formats)
        const avail = res.formats.filter((f) => f.available).map((f) => f.name)
        const defaults = res.defaults.filter((d) => avail.includes(d))
        setSelected(defaults.length ? defaults : avail.slice(0, 1))
      })
      .catch((err) => message.error(err.message ?? '读取导出格式失败'))
  }, [])

  // 轮询正在进行的导出任务
  useEffect(() => {
    if (!job || !isActiveStatus(job.status)) return
    const t = window.setInterval(async () => {
      try {
        const res = await api.deployJob(job.id)
        setJob(res.job)
        if (!isActiveStatus(res.job.status)) {
          const r = await api.deployResult(job.id)
          setResult(r.result)
          message.success('导出已结束')
          onChanged()
        }
      } catch {
        // 忽略单次轮询失败
      }
    }, 2000)
    return () => window.clearInterval(t)
  }, [job?.id, job?.status, onChanged])

  const start = async () => {
    if (!selected.length) {
      message.warning('请至少选择一个导出格式')
      return
    }
    setStarting(true)
    setResult(null)
    setVerify(null)
    try {
      const res = await api.startDeploy({ model_id: modelId, formats: selected, imgsz })
      setJob(res.job)
      message.success('导出已启动')
    } catch (err: any) {
      message.error(err.message ?? '启动导出失败')
    } finally {
      setStarting(false)
    }
  }

  const doVerify = async () => {
    if (!job) return
    setVerifying(true)
    try {
      const res = await api.verifyDeploy(job.id)
      setVerify(res.verify)
      message[res.verify.ok ? 'success' : 'warning'](
        res.verify.ok
          ? `校验通过（${res.verify.checked} 个产物）`
          : `发现 ${res.verify.problems.length} 个问题`,
      )
    } catch (err: any) {
      message.error(err.message ?? '校验失败')
    } finally {
      setVerifying(false)
    }
  }

  const availableFormats = formats.filter((f) => f.available)
  const unavailableFormats = formats.filter((f) => !f.available)

  return (
    <>
      <Card title="导出部署" size="small">
        <Paragraph type="secondary">
          导出跑在独立子进程，逐个格式导出、互不影响；不可用的格式在启动前就被拒绝并说明原因。
          产物会复制到导出目录并记录大小与 sha256，可随时校验完整性。
        </Paragraph>

        <Space direction="vertical" style={{ width: '100%' }} size={12}>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              可用格式（{availableFormats.length}）
            </Text>
            <div style={{ marginTop: 6 }}>
              <Space wrap size={4}>
                {availableFormats.map((f) => {
                  const on = selected.includes(f.name)
                  return (
                    <Tag
                      key={f.name}
                      color={on ? 'cyan' : undefined}
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() =>
                        setSelected(on ? selected.filter((x) => x !== f.name) : [...selected, f.name])
                      }
                    >
                      {f.label}
                    </Tag>
                  )
                })}
                {availableFormats.length === 0 && (
                  <Text type="secondary">没有可用格式（依赖未安装）</Text>
                )}
              </Space>
            </div>
          </div>

          {unavailableFormats.length > 0 && (
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                不可用格式（鼠标悬停看原因）
              </Text>
              <div style={{ marginTop: 6 }}>
                <Space wrap size={4}>
                  {unavailableFormats.map((f) => (
                    <Tooltip key={f.name} title={`${f.reason}${f.needs_gpu ? '（需要 GPU）' : ''}`}>
                      <Tag color="default" style={{ opacity: 0.6 }}>
                        {f.label}
                      </Tag>
                    </Tooltip>
                  ))}
                </Space>
              </div>
            </div>
          )}

          <Space wrap>
            <span>
              <Text type="secondary" style={{ fontSize: 12, marginRight: 6 }}>
                imgsz
              </Text>
              <InputNumber min={16} value={imgsz} onChange={(v) => setImgsz(Number(v) || 640)} />
            </span>
            <Button type="primary" icon={<CloudDownloadOutlined />} loading={starting} onClick={start}>
              开始导出
            </Button>
            <Text type="secondary" style={{ fontSize: 12 }}>
              默认只导出生态最通用的格式；ONNX 需要本机已安装 onnx
            </Text>
          </Space>
        </Space>

        {job && (
          <Alert
            style={{ marginTop: 12 }}
            type={job.status === 'finished' ? 'success' : job.status === 'failed' ? 'error' : 'info'}
            showIcon
            message={
              <Space>
                <span>导出任务 {job.id}</span>
                <StatusTag status={job.status} label={job.status_label} />
              </Space>
            }
            description={job.error || job.message || undefined}
            action={
              job && !isActiveStatus(job.status) ? (
                <Button size="small" icon={<SafetyCertificateOutlined />} loading={verifying} onClick={doVerify}>
                  校验产物
                </Button>
              ) : null
            }
          />
        )}
      </Card>

      {result && (
        <Card title="本次导出产物" size="small" style={{ marginTop: 16 }}>
          <Table
            size="small"
            rowKey="format"
            pagination={false}
            dataSource={result.artifacts}
            columns={[
              { title: '格式', dataIndex: 'format', width: 120 },
              {
                title: '结果',
                dataIndex: 'ok',
                width: 90,
                render: (ok: boolean) => (ok ? <Tag color="success">成功</Tag> : <Tag color="error">失败</Tag>),
              },
              { title: '文件名', dataIndex: 'name', render: (v: string, r) => v || (r.is_dir ? '（目录产物）' : '-') },
              {
                title: '大小',
                dataIndex: 'size',
                width: 110,
                render: (v: number) => formatBytes(v),
              },
              {
                title: 'sha256',
                dataIndex: 'sha256',
                width: 150,
                render: (v: string) =>
                  v ? (
                    <Text className="mono" style={{ fontSize: 11 }} title={v}>
                      {v.slice(0, 12)}…
                    </Text>
                  ) : (
                    <Text type="secondary">-</Text>
                  ),
              },
              {
                title: '说明',
                dataIndex: 'error',
                render: (v: string) => v || <Text type="secondary">-</Text>,
              },
              {
                title: '下载',
                width: 80,
                render: (_, r) =>
                  r.ok ? (
                    <a href={deployDownloadUrl(job!.id, r.format)} target="_blank" rel="noreferrer">
                      下载
                    </a>
                  ) : (
                    <Text type="secondary">-</Text>
                  ),
              },
            ]}
          />
          {verify && (
            <Alert
              style={{ marginTop: 12 }}
              type={verify.ok ? 'success' : 'warning'}
              showIcon
              message={
                verify.ok
                  ? `完整性校验通过（检查 ${verify.checked} 个产物）`
                  : `发现 ${verify.problems.length} 个完整性问题`
              }
              description={
                verify.problems.length ? (
                  <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
                    {verify.problems.map((p, i) => (
                      <li key={i}>
                        {p.format}: {p.issue}（{p.path}）
                      </li>
                    ))}
                  </ul>
                ) : undefined
              }
            />
          )}
        </Card>
      )}

      {card.deploys.length > 0 && (
        <Card title="历史导出记录" size="small" style={{ marginTop: 16 }}>
          <Collapse
            size="small"
            items={card.deploys
              .slice()
              .reverse()
              .map((d) => ({
                key: String(d.deploy_id),
                label: (
                  <Space>
                    <Text>{d.deploy_id}</Text>
                    <Tag color={d.ok ? 'success' : 'error'}>{d.ok ? '成功' : '失败'}</Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {d.created_at}
                    </Text>
                  </Space>
                ),
                children: (
                  <Space wrap>
                    {(d.artifacts ?? []).map((a: any, i: number) => (
                      <Tag key={i} color={a.ok ? 'cyan' : 'red'}>
                        {a.format} {a.ok ? formatBytes(a.size) : '失败'}
                      </Tag>
                    ))}
                  </Space>
                ),
              }))}
          />
        </Card>
      )}
    </>
  )
}
