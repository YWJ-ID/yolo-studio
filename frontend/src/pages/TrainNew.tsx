import { useEffect, useState } from 'react'
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd'
import { ArrowLeftOutlined, PlayCircleOutlined, UploadOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { BaseWeightItem, DatasetVersion, EnvResponse, TrainBackendsResponse } from '../types'

const { Text, Paragraph } = Typography

/**
 * 内置候选：接口暂时不可用时的兜底。正常情况下候选来自
 * `GET /api/train/weights`（含权重目录、模型库、历史训练产物，并可上传）。
 */
const WEIGHTS_BY_TASK: Record<string, { value: string; label: string }[]> = {
  detect: [
    { value: 'yolov8n.yaml', label: 'yolov8n.yaml（结构文件，从零训练）' },
    { value: 'yolov8s.yaml', label: 'yolov8s.yaml（结构文件，从零训练）' },
    { value: 'yolov8m.yaml', label: 'yolov8m.yaml（结构文件，从零训练）' },
    { value: 'yolov8n.pt', label: 'yolov8n.pt（预训练权重，需联网下载）' },
    { value: 'yolov8s.pt', label: 'yolov8s.pt（预训练权重，需联网下载）' },
  ],
  classify: [
    { value: 'yolov8-cls.yaml', label: 'yolov8-cls.yaml（结构文件，从零训练）' },
    { value: 'yolov8n-cls.pt', label: 'yolov8n-cls.pt（预训练权重，需联网下载）' },
  ],
}

/** 候选权重的来源排序：内置在前，历史产物在后。 */
const WEIGHT_SOURCE_ORDER: Record<string, number> = {
  builtin: 0,
  weights_dir: 1,
  model_library: 2,
  run: 3,
}

/** 三态布尔：留空 = 用框架默认，避免把「没设置」写成 false 覆盖掉框架默认值。 */
const TRI_OPTIONS = [
  { value: '', label: '默认' },
  { value: 'true', label: '开启' },
  { value: 'false', label: '关闭' },
]

const numOrNull = (v: unknown) => (v === undefined || v === null || v === '' ? null : Number(v))
const boolOrNull = (v: unknown) => (v === 'true' ? true : v === 'false' ? false : null)
const strOrNull = (v: unknown) => (v === undefined || v === null || v === '' ? null : String(v))


/** 新建训练（M5-07 的新建部分）。 */
export default function TrainNew() {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [env, setEnv] = useState<EnvResponse | null>(null)
  const [backends, setBackends] = useState<TrainBackendsResponse | null>(null)
  const [versions, setVersions] = useState<DatasetVersion[]>([])
  const [weights, setWeights] = useState<BaseWeightItem[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [task, setTask] = useState('detect')

  useEffect(() => {
    Promise.all([
      api.env(),
      api.trainBackends(),
      api.versions(),
      api.trainWeights().catch(() => null),
    ])
      .then(([e, b, v, w]) => {
        setEnv(e)
        setBackends(b)
        setVersions(v)
        setWeights(w?.weights ?? [])
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

  // 候选按任务类型过滤（task 为空的候选不限任务）；接口不可用时退回内置候选
  const available = weights.filter((w) => !w.task || w.task === task)
  const weightOptions = available.length
    ? available
        .slice()
        .sort((a, b) => (WEIGHT_SOURCE_ORDER[a.source] ?? 9) - (WEIGHT_SOURCE_ORDER[b.source] ?? 9))
        .map((w) => ({ value: w.value, label: w.label }))
    : WEIGHTS_BY_TASK[task] ?? WEIGHTS_BY_TASK.detect

  const doUpload = async (file: File, overwrite: boolean) => {
    setUploading(true)
    try {
      const res = await api.uploadTrainWeight(file, overwrite)
      message.success(`已上传 ${res.name}（${(res.size_bytes / 1024 / 1024).toFixed(1)} MB）`)
      form.setFieldValue('weights', res.path)
      const list = await api.trainWeights().catch(() => null)
      if (list) setWeights(list.weights)
    } catch (err: any) {
      const msg = String(err?.message ?? '上传失败')
      if (!overwrite && msg.includes('已存在')) {
        Modal.confirm({
          title: '权重已存在',
          content: `${file.name} 已存在于权重目录，是否覆盖？`,
          okText: '覆盖',
          cancelText: '取消',
          onOk: () => doUpload(file, true),
        })
      } else {
        message.error(msg)
      }
    } finally {
      setUploading(false)
    }
  }

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
        lr0: numOrNull(values.lr0),
        lrf: numOrNull(values.lrf),
        cos_lr: boolOrNull(values.cos_lr),
        warmup_epochs: numOrNull(values.warmup_epochs),
        label_smoothing: numOrNull(values.label_smoothing),
        dropout: numOrNull(values.dropout),
        freeze: numOrNull(values.freeze),
        close_mosaic: numOrNull(values.close_mosaic),
        mosaic: numOrNull(values.mosaic),
        mixup: numOrNull(values.mixup),
        fliplr: numOrNull(values.fliplr),
        hsv_h: numOrNull(values.hsv_h),
        hsv_s: numOrNull(values.hsv_s),
        hsv_v: numOrNull(values.hsv_v),
        scale: numOrNull(values.scale),
        translate: numOrNull(values.translate),
        rect: boolOrNull(values.rect),
        cache: strOrNull(values.cache),
        amp: boolOrNull(values.amp),
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
                label="基础模型（权重）"
                tooltip=".yaml 结构文件从零训练（离线、不下载）；.pt 为预训练权重。可选权重目录 / 模型库 / 历史训练产物，或上传自定义权重；留空按任务取默认 .yaml"
              >
                <AutoComplete
                  options={weightOptions}
                  placeholder={task === 'classify' ? '留空 = yolov8-cls.yaml' : '留空 = yolov8n.yaml'}
                  allowClear
                />
              </Form.Item>
              <Space size={8} style={{ marginTop: -12, marginBottom: 12 }}>
                <Upload
                  accept=".pt,.yaml,.yml"
                  showUploadList={false}
                  beforeUpload={(file) => {
                    doUpload(file as unknown as File, false)
                    return false
                  }}
                  disabled={uploading}
                >
                  <Button size="small" icon={<UploadOutlined />} loading={uploading}>
                    上传自定义权重
                  </Button>
                </Upload>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  支持 .pt / .yaml，上传到权重目录后即可选择
                </Text>
              </Space>
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

          <Collapse
            ghost
            style={{ marginBottom: 8 }}
            items={[
              {
                key: 'strategy',
                label: '训练策略与正则（留空 = 用 ultralytics 默认）',
                children: (
                  <Row gutter={16}>
                    <Col xs={12} md={6}>
                      <Form.Item name="lrf" label="最终学习率系数 lrf" tooltip="相对 lr0，例如 0.01">
                        <InputNumber min={0} step={0.01} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="cos_lr" label="余弦学习率调度">
                        <Select options={TRI_OPTIONS} placeholder="默认" allowClear />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="warmup_epochs" label="warmup 轮数">
                        <InputNumber min={0} step={0.5} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="close_mosaic" label="末 N 轮关闭 mosaic" tooltip="训练最后 N 轮停用 mosaic，通常能涨点">
                        <InputNumber min={0} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="label_smoothing" label="标签平滑">
                        <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="dropout" label="dropout（分类任务）">
                        <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="freeze" label="冻结前 N 层" tooltip="迁移学习常用，0 表示不冻结">
                        <InputNumber min={0} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                  </Row>
                ),
              },
              {
                key: 'augment',
                label: '数据增强（留空 = 用 ultralytics 默认）',
                children: (
                  <Row gutter={16}>
                    <Col xs={12} md={6}>
                      <Form.Item name="mosaic" label="mosaic">
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="mixup" label="mixup">
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="fliplr" label="水平翻转 fliplr">
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="scale" label="缩放 scale">
                        <InputNumber min={0} step={0.1} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="translate" label="平移 translate">
                        <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={8} md={4}>
                      <Form.Item name="hsv_h" label="HSV-H">
                        <InputNumber min={0} max={1} step={0.005} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={8} md={4}>
                      <Form.Item name="hsv_s" label="HSV-S">
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                    <Col xs={8} md={4}>
                      <Form.Item name="hsv_v" label="HSV-V">
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} placeholder="默认" />
                      </Form.Item>
                    </Col>
                  </Row>
                ),
              },
              {
                key: 'engine',
                label: '工程权衡（留空 = 用 ultralytics 默认）',
                children: (
                  <Row gutter={16}>
                    <Col xs={12} md={6}>
                      <Form.Item name="rect" label="矩形训练 rect">
                        <Select options={TRI_OPTIONS} placeholder="默认" allowClear />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="cache" label="数据缓存">
                        <Select
                          options={[
                            { value: '', label: '不缓存' },
                            { value: 'ram', label: '内存 ram' },
                            { value: 'disk', label: '磁盘 disk' },
                          ]}
                          placeholder="默认"
                          allowClear
                        />
                      </Form.Item>
                    </Col>
                    <Col xs={12} md={6}>
                      <Form.Item name="amp" label="混合精度 amp" tooltip="仅 GPU 有意义；CPU 自动忽略">
                        <Select options={TRI_OPTIONS} placeholder="默认" allowClear />
                      </Form.Item>
                    </Col>
                  </Row>
                ),
              },
            ]}
          />

          <Form.Item name="name" label="任务名" tooltip="同时作为训练目录名，留空自动生成">
            <Input style={{ maxWidth: 360 }} placeholder="例如 dms_yolov8n_v1" allowClear />
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
