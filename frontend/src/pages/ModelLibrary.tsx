import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Empty,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { BarChartOutlined, EyeOutlined, ReloadOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ModelSummary } from '../types'
import { formatMetric } from '../utils'

const { Text, Paragraph } = Typography

const TASK_LABEL: Record<string, { text: string; color: string }> = {
  detect: { text: '目标检测', color: 'blue' },
  classify: { text: '图像分类', color: 'purple' },
}

function bestMetric(m: ModelSummary): string {
  const nested = (m.best?.best ?? {}) as Record<string, number>
  const key = ['mAP50-95', 'mAP50', 'top1'].find((k) => nested[k] != null)
  if (!key) return '-'
  return `${key} ${formatMetric(nested[key])}`
}

/** 模型库列表（M5-08）。 */
export default function ModelLibrary() {
  const navigate = useNavigate()
  const [models, setModels] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<string[]>([])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.models()
      setModels(res.models)
      setError('')
    } catch (err: any) {
      setError(err.message ?? '读取模型库失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const goCompare = () => {
    if (selected.length < 2) {
      message.warning('至少选择两个模型才能对比')
      return
    }
    navigate(`/models/compare?ids=${selected.map(encodeURIComponent).join(',')}`)
  }

  return (
    <div className="page">
      <Card
        title="模型库"
        size="small"
        extra={
          <Space>
            <Button
              size="small"
              icon={<BarChartOutlined />}
              disabled={selected.length < 2}
              onClick={goCompare}
            >
              对比所选（{selected.length}）
            </Button>
            <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
          </Space>
        }
      >
        <Paragraph type="secondary">
          训练成功会自动注册到模型库并触发评估。模型卡只记录权重路径与评估目录的
          <Text strong>引用</Text>，指标始终从评估结果文件读取，不会出现两处数据不一致。
        </Paragraph>

        {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}

        {models.length === 0 && !loading ? (
          <Empty description="还没有模型，训练完成后会自动注册；也可在训练列表点「注册」" />
        ) : (
          <Table
            size="small"
            rowKey="model_id"
            loading={loading}
            dataSource={models}
            pagination={false}
            scroll={{ x: 'max-content' }}
            rowSelection={{
              selectedRowKeys: selected,
              onChange: (keys) => setSelected(keys as string[]),
            }}
            columns={[
              {
                title: '模型',
                dataIndex: 'name',
                render: (name: string, r) => (
                  <Space direction="vertical" size={0}>
                    <a onClick={() => navigate(`/models/${encodeURIComponent(r.model_id)}`)}>
                      <Text strong>{name || r.model_id}</Text>
                    </a>
                    <Text type="secondary" className="mono" style={{ fontSize: 11 }}>
                      {r.model_id}
                    </Text>
                  </Space>
                ),
              },
              {
                title: '任务',
                dataIndex: 'task',
                width: 100,
                render: (t: string) => {
                  const info = TASK_LABEL[t]
                  return info ? <Tag color={info.color}>{info.text}</Tag> : <Tag>{t}</Tag>
                },
              },
              {
                title: '类别',
                dataIndex: 'num_classes',
                width: 70,
                render: (n: number, r) => (
                  <span title={r.classes.join('、')}>{n}</span>
                ),
              },
              {
                title: '数据集',
                dataIndex: 'dataset_name',
                width: 180,
                render: (v: string) => v || <Text type="secondary">-</Text>,
              },
              {
                title: '评估划分',
                dataIndex: 'eval_splits',
                width: 130,
                render: (splits: string[], r) =>
                  splits.length ? (
                    <Space size={4}>
                      {splits.map((s) => (
                        <Tag key={s} color="geekblue">
                          {s}
                        </Tag>
                      ))}
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        ×{r.num_evals}
                      </Text>
                    </Space>
                  ) : (
                    <Text type="secondary">未评估</Text>
                  ),
              },
              {
                title: '最佳指标',
                width: 160,
                render: (_, r) => bestMetric(r),
              },
              {
                title: '导出格式',
                dataIndex: 'deploy_formats',
                width: 160,
                render: (fmts: string[]) =>
                  fmts.length ? (
                    <Space size={4} wrap>
                      {fmts.map((f) => (
                        <Tag key={f} color="cyan">
                          {f}
                        </Tag>
                      ))}
                    </Space>
                  ) : (
                    <Text type="secondary">未导出</Text>
                  ),
              },
              {
                title: '操作',
                width: 90,
                render: (_, r) => (
                  <Button
                    size="small"
                    type="link"
                    icon={<EyeOutlined />}
                    onClick={() => navigate(`/models/${encodeURIComponent(r.model_id)}`)}
                  >
                    详情
                  </Button>
                ),
              },
            ]}
          />
        )}
      </Card>
    </div>
  )
}
