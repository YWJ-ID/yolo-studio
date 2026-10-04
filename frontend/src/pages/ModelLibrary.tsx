import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Empty,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  BarChartOutlined,
  EyeOutlined,
  ImportOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ModelSummary } from '../types'
import { formatMetric } from '../utils'

const { Text, Paragraph } = Typography

const TASK_LABEL: Record<string, { text: string; color: string }> = {
  detect: { text: '目标检测', color: 'blue' },
  classify: { text: '图像分类', color: 'purple' },
  segment: { text: '实例分割', color: 'geekblue' },
}

interface ImportValues {
  weights: string
  data_yaml?: string
  name?: string
  task: string
  classesText?: string
  imgsz: number
  batch: number
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
  const [importForm] = Form.useForm<ImportValues>()
  const [models, setModels] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const [importOpen, setImportOpen] = useState(false)
  const [importing, setImporting] = useState(false)

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

  const onImport = async (values: ImportValues) => {
    if (!values.weights?.trim()) {
      message.warning('请填写 .pt 权重路径')
      return
    }
    setImporting(true)
    try {
      const res = await api.importModel({
        weights: values.weights.trim(),
        data_yaml: values.data_yaml?.trim() ?? '',
        name: values.name?.trim() ?? '',
        task: values.task,
        classes: (values.classesText ?? '')
          .split(/[,;|、\n\r\t]+/)
          .map((s) => s.trim())
          .filter(Boolean),
        imgsz: values.imgsz,
        batch: values.batch,
      })
      message.success(`已导入：${res.model.name || res.model.model_id}`)
      setImportOpen(false)
      importForm.resetFields()
      await load()
      navigate(`/models/${encodeURIComponent(res.model.model_id)}`)
    } catch (err: any) {
      message.error(err.message ?? '导入失败')
    } finally {
      setImporting(false)
    }
  }

  return (
    <div className="page">
      <Card
        title="模型库"
        size="small"
        extra={
          <Space>
            <Button size="small" icon={<ImportOutlined />} onClick={() => setImportOpen(true)}>
              导入外部模型
            </Button>
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
          非本项目的训练权重可用「导入外部模型」登记（目前仅支持 <Text code>.pt</Text>）。
        </Paragraph>

        {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}

        {models.length === 0 && !loading ? (
          <Empty description="还没有模型，训练完成后会自动注册；也可导入外部 .pt 权重" />
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
                    <Space size={6}>
                      <a onClick={() => navigate(`/models/${encodeURIComponent(r.model_id)}`)}>
                        <Text strong>{name || r.model_id}</Text>
                      </a>
                      {r.source === 'external' && (
                        <Tag color="orange" title="非本项目训练任务，由外部 .pt 权重导入">
                          外部
                        </Tag>
                      )}
                    </Space>
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
                render: (n: number, r) => <span title={r.classes.join('、')}>{n}</span>,
              },
              {
                title: '数据集',
                dataIndex: 'dataset_name',
                width: 180,
                render: (v: string, r) =>
                  v || (r.has_data_yaml ? <Text type="secondary">-</Text> : <Text type="secondary">无</Text>),
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

      <Modal
        open={importOpen}
        title="导入外部模型"
        okText="导入"
        cancelText="取消"
        confirmLoading={importing}
        width={620}
        onCancel={() => setImportOpen(false)}
        onOk={() => importForm.submit()}
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="只支持 .pt 权重"
          description="导入后即可对它发起评估、对比与导出。评估需要配套的 data.yaml（带真值标签）；不填也能导入，但导入后无法评估（导出仍然可用）。"
        />
        <Form
          form={importForm}
          layout="vertical"
          onFinish={onImport}
          initialValues={{ task: 'detect', imgsz: 640, batch: 16 }}
        >
          <Form.Item
            name="weights"
            label=".pt 权重路径"
            tooltip="服务器本机上的绝对路径；登记的是引用，请勿移动该文件"
            rules={[{ required: true, message: '请填写权重路径' }]}
          >
            <Input placeholder="例如 D:\models\best.pt" allowClear />
          </Form.Item>
          <Form.Item
            name="data_yaml"
            label="数据集 data.yaml（可选，但评估需要）"
            tooltip="指向带真值标注的数据集；类别会自动从其中的 names 读取"
          >
            <Input placeholder="例如 D:\datasets\mine\data.yaml" allowClear />
          </Form.Item>
          <Form.Item name="name" label="展示名（可选）">
            <Input placeholder="默认用权重文件名" allowClear />
          </Form.Item>
          <Space size="large" wrap>
            <Form.Item name="task" label="任务类型" style={{ minWidth: 150 }}>
              <Select
                options={[
                  { value: 'detect', label: '目标检测' },
                  { value: 'classify', label: '图像分类' },
                  { value: 'segment', label: '实例分割' },
                ]}
              />
            </Form.Item>
            <Form.Item name="imgsz" label="imgsz" style={{ width: 120 }}>
              <InputNumber style={{ width: '100%' }} min={32} max={1920} step={32} />
            </Form.Item>
            <Form.Item name="batch" label="batch" style={{ width: 120 }}>
              <InputNumber style={{ width: '100%' }} min={1} max={512} />
            </Form.Item>
          </Space>
          <Form.Item name="classesText" label="类别清单（可选，逗号分隔）">
            <Input placeholder="留空则读取 data.yaml 的 names" allowClear />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
