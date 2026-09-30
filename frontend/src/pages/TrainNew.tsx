import { useEffect, useState } from 'react'
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Tag,
  Typography,
  message,
} from 'antd'
import { ArrowLeftOutlined, PlayCircleOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { DatasetVersion, EnvResponse, TrainBackendsResponse } from '../types'

const { Text, Paragraph } = Typography

const WEIGHTS_BY_TASK: Record<string, { value: string; label: string }[]> = {
  detect: [
    { value: 'yolo11n.yaml', label: 'yolo11n.yaml（结构文件，从零训练）' },
    { value: 'yolo11s.yaml', label: 'yolo11s.yaml' },
    { value: 'yolo11m.yaml', label: 'yolo11m.yaml' },
    { value: 'yolo11n.pt', label: 'yolo11n.pt（预训练权重）' },
  ],
  classify: [
    { value: 'yolo11-cls.yaml', label: 'yolo11-cls.yaml（结构文件，从零训练）' },
    { value: 'yolo11n-cls.pt', label: 'yolo11n-cls.pt（预训练权重）' },
  ],
}

/** 新建训练（M5-07 的新建部分）。 */
export default function TrainNew() {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [env, setEnv] = useState<EnvResponse | null>(null)
  const [backends, setBackends] = useState<TrainBackendsResponse | null>(null)
  const [versions, setVersions] = useState<DatasetVersion[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [task, setTask] = useState('detect')

  useEffect(() => {
    Promise.all([api.env(), api.trainBackends(), api.versions()])
      .then(([e, b, v]) => {
        setEnv(e)
        setBackends(b)
        setVersions(v)
        form.setFieldsValue({
          device: b.defaults.device || 'cpu',
          task: 'detect',
        })
      })
      .catch((err) => message.error(err.message ?? '读取环境失败'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const dataYamlOptions = versions.map((v) => ({
    value: `${v.path}\\data.yaml`,
    label: `${v.name}（${v.task}，train ${v.images?.train ?? 0} / val ${v.images?.val ?? 0} / test ${v.images?.test ?? 0}）`,
  }))

  const onSubmit = async (values: any) => {
    if (!values.data_yaml?.trim()) {
      message.warning('请选择或填写 data.yaml 路径')
      return
    }
    setSubmitting(true)
    try {
      const res = await api.startTrain({
        data_yaml: values.data_yaml.trim(),
        weights: values.weights?.trim() || '',
        task: values.task,
        epochs: values.epochs,
        imgsz: values.imgsz,
        batch: values.batch,
        device: values.device,
        workers: values.workers,
        seed: values.seed,
        patience: values.patience,
        optimizer: values.optimizer,
        lr0: values.lr0 ?? null,
        name: values.name?.trim() || null,
      })
      message.success('训练已启动')
      navigate(`/train/${encodeURIComponent(res.job.id)}`)
    } catch (err: any) {
      message.error(err.message ?? '启动训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  const noCuda = env && !env.cuda_available

  return (
    <div className="page">
      <Card
        title="新建训练"
        size="small"
        extra={
          <Button size="small" icon={<ArrowLeftOutlined />} onClick={() => navigate('/train')}>
            返回列表
          </Button>
        }
      >
        {noCuda && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 12 }}
            message="本机没有可用的 CUDA，训练将以 CPU 运行"
            description="CPU 训练明显更慢。先用小 imgsz（如 64/128）、少 epoch、小 batch 验证流程，确认无误后再放大。"
          />
        )}

        <Form
          form={form}
          layout="vertical"
          onFinish={onSubmit}
          initialValues={{
            epochs: 100,
            imgsz: 640,
            batch: 16,
            workers: 0,
            seed: 42,
            patience: 100,
            optimizer: 'auto',
          }}
        >
          <Row gutter={16}>
            <Col xs={24} lg={14}>
              <Form.Item
                name="data_yaml"
                label="数据集 data.yaml"
                tooltip="来自「数据导入」生成的数据集版本，也可直接填任意 data.yaml 的绝对路径"
                rules={[{ required: true, message: '请填写 data.yaml 路径' }]}
              >
                <AutoComplete
                  options={dataYamlOptions}
                  placeholder="选择已生成的数据集，或输入 data.yaml 绝对路径"
                  filterOption={(input, option) =>
                    String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                  }
                />
              </Form.Item>
            </Col>
            <Col xs={24} lg={10}>
              <Form.Item
                name="task"
                label="任务类型"
                rules={[{ required: true }]}
              >
                <Select
                  onChange={(v) => {
                    setTask(v)
                    form.setFieldValue('weights', '')
                  }}
                  options={[
                    { value: 'detect', label: '目标检测（detect）' },
                    { value: 'classify', label: '图像分类（classify）' },
                  ]}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} lg={14}>
              <Form.Item
                name="weights"
                label="权重"
                tooltip=".yaml 表示从结构文件从零训练（离线可用，不需下载）；.pt 为预训练权重。留空按任务取默认 .yaml"
              >
                <AutoComplete
                  options={WEIGHTS_BY_TASK[task] ?? WEIGHTS_BY_TASK.detect}
                  placeholder={task === 'classify' ? '留空 = yolo11-cls.yaml' : '留空 = yolo11n.yaml'}
                  allowClear
                />
              </Form.Item>
            </Col>
            <Col xs={24} lg={10}>
              <Form.Item name="device" label="训练设备" tooltip="cpu | cuda:0；本机无 GPU 时保持 cpu">
                <Select
                  options={(env?.devices?.length ? env.devices : ['cpu']).map((d) => ({
                    value: d,
                    label: d,
                  }))}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={12} md={6}>
              <Form.Item name="epochs" label="epochs">
                <InputNumber min={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item name="imgsz" label="imgsz">
                <InputNumber min={16} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item name="batch" label="batch">
                <InputNumber min={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item name="workers" label="dataloader workers" tooltip="CPU 训练建议 0">
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={12} md={6}>
              <Form.Item name="seed" label="随机种子">
                <InputNumber style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item name="patience" label="早停 patience">
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item name="optimizer" label="优化器">
                <Select
                  options={[
                    { value: 'auto', label: 'auto' },
                    { value: 'SGD', label: 'SGD' },
                    { value: 'Adam', label: 'Adam' },
                    { value: 'AdamW', label: 'AdamW' },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item name="lr0" label="初始学习率 lr0" tooltip="留空由 ultralytics 自动决定">
                <InputNumber min={0} step={0.001} style={{ width: '100%' }} placeholder="自动" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="name" label="任务名" tooltip="同时作为训练目录名，留空自动生成">
            <Input style={{ maxWidth: 360 }} placeholder="例如 dms_yolo11n_v1" allowClear />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<PlayCircleOutlined />}
                loading={submitting}
              >
                启动训练
              </Button>
              <Text type="secondary">启动后自动跳到详情页，可实时看曲线、资源与日志</Text>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card title="运行环境" size="small" style={{ marginTop: 16 }}>
        <Descriptions column={{ xs: 1, md: 2 }} size="small">
          <Descriptions.Item label="ultralytics">
            {env?.ultralytics ? <Tag color="green">{env.ultralytics}</Tag> : <Tag>未安装</Tag>}
          </Descriptions.Item>
          <Descriptions.Item label="PyTorch">{env?.torch ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="CUDA">
            {env?.cuda_available ? (
              <Tag color="green">可用（{env.cuda_device_count} 卡）</Tag>
            ) : (
              <Tag color="orange">不可用</Tag>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="训练解释器">
            <Text className="mono" style={{ fontSize: 11 }}>
              {env?.training_python ?? '-'}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="训练根目录">
            <Text className="mono" style={{ fontSize: 11 }}>
              {backends?.defaults.runs_dir ?? '-'}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="资源监控">
            <Space size={4}>
              <Tag color={backends?.monitor.cpu_monitor ? 'green' : 'orange'}>CPU</Tag>
              <Tag color={backends?.monitor.gpu_monitor ? 'green' : 'orange'}>
                GPU{backends?.monitor.gpu_count ? `×${backends.monitor.gpu_count}` : ''}
              </Tag>
            </Space>
          </Descriptions.Item>
        </Descriptions>
        {backends?.monitor.gpu_error && !backends.monitor.gpu_monitor && (
          <Paragraph type="secondary" style={{ marginBottom: 0, marginTop: 8, fontSize: 12 }}>
            GPU 监控不可用：{backends.monitor.gpu_error}
          </Paragraph>
        )}
      </Card>
    </div>
  )
}
