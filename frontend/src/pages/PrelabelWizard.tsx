import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  InputNumber,
  Progress,
  Row,
  Select,
  Slider,
  Space,
  Statistic,
  Steps,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  CheckCircleOutlined,
  ExperimentOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  TagsOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { api } from '../api/client'
import type {
  ExportReport,
  InferFormatInfo,
  InferWeightItem,
  PrelabelJob,
  SampleImage,
} from '../types'
import { SampleGrid } from '../components/SampleGrid'

const { Text, Paragraph } = Typography

/** 分隔符支持逗号、分号、竖线、顿号、换行（与实时验证页一致）。 */
function parseClassList(text: string): string[] {
  return text
    .split(/[,;|、\n\r\t]+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

interface ConfigValues {
  images_dir: string
  classesText?: string
  task: string
  device: string
  imgsz: number
  conf: number
  iou: number
  maxDet: number
  limit: number
}

interface ExportValues {
  out_name: string
  task: string
  name_style: string
  file_mode: string
  overwrite: string
  splitEnabled: string
  trainRatio: number
  valRatio: number
  seed: number
  stratified: string
  respectGroups: string
  cleanEnabled: string
  cleanNear: string
}

/** M7 预标注：权重 + 图片目录 → 预览 → 确认 → 生成数据集。 */
export default function PrelabelWizard() {
  const [form] = Form.useForm<ConfigValues>()
  const [exportForm] = Form.useForm<ExportValues>()

  const [formats, setFormats] = useState<InferFormatInfo[]>([])
  const [weights, setWeights] = useState<InferWeightItem[]>([])
  const [weightsValue, setWeightsValue] = useState('')
  const [jobs, setJobs] = useState<PrelabelJob[]>([])

  const [step, setStep] = useState(0)
  const [job, setJob] = useState<PrelabelJob | null>(null)
  const [samples, setSamples] = useState<SampleImage[]>([])
  const [sampleCategories, setSampleCategories] = useState<string[]>([])
  const [starting, setStarting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exportReport, setExportReport] = useState<{ out_dir: string; report: ExportReport } | null>(null)

  const timerRef = useRef<number | null>(null)

  const loadJobs = useCallback(() => {
    api
      .prelabelJobs()
      .then((res) => setJobs(res.jobs))
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    api
      .prelabelFormats()
      .then((res) => setFormats(res.formats))
      .catch((err) => message.error(err.message ?? '读取推理格式失败'))
    api
      .prelabelWeights()
      .then((res) => setWeights(res.weights))
      .catch(() => undefined)
    loadJobs()
  }, [loadJobs])

  const stopPolling = useCallback(() => {
    if (timerRef.current != null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  useEffect(() => () => stopPolling(), [stopPolling])

  const refreshSamples = useCallback(async (id: string) => {
    try {
      const res = await api.prelabelSamples(id, 0, 12)
      setSamples(res.images)
      setSampleCategories(res.categories)
    } catch {
      setSamples([])
    }
  }, [])

  const startPolling = useCallback(
    (id: string) => {
      stopPolling()
      timerRef.current = window.setInterval(async () => {
        try {
          const res = await api.prelabelJob(id)
          setJob(res.job)
          if (res.job.status === 'finished' || res.job.status === 'failed') {
            stopPolling()
            loadJobs()
            if (res.job.status === 'finished') {
              void refreshSamples(id)
            }
          }
        } catch {
          stopPolling()
        }
      }, 1500)
    },
    [stopPolling, loadJobs, refreshSamples],
  )

  const formatOf = useCallback(
    (path: string) => weights.find((w) => w.value === path)?.format ?? '',
    [weights],
  )
  const selectedFormat = formats.find((f) => f.name === formatOf(weightsValue))

  const onStart = async (values: ConfigValues) => {
    if (!weightsValue.trim()) {
      message.warning('请先选择或填写权重路径')
      return
    }
    if (!values.images_dir?.trim()) {
      message.warning('请填写待标注的图片目录')
      return
    }
    setStarting(true)
    try {
      const res = await api.startPrelabel({
        weights: weightsValue.trim(),
        images_dir: values.images_dir.trim(),
        classes: parseClassList(values.classesText ?? ''),
        task: values.task,
        device: values.device,
        imgsz: values.imgsz,
        conf: values.conf,
        iou: values.iou,
        max_det: values.maxDet,
        fmt: formatOf(weightsValue.trim()),
        limit: values.limit,
      })
      setJob(res.job)
      setSamples([])
      setExportReport(null)
      setStep(1)
      startPolling(res.job.id)
      loadJobs()
      message.success('已开始预标注，正在后台推理…')
    } catch (err: any) {
      message.error(err.message ?? '启动预标注失败')
    } finally {
      setStarting(false)
    }
  }

  const reset = () => {
    stopPolling()
    setJob(null)
    setSamples([])
    setExportReport(null)
    setStep(0)
  }

  const watchJob = (item: PrelabelJob) => {
    setJob(item)
    setExportReport(null)
    setStep(1)
    if (item.status === 'running' || item.status === 'pending') {
      startPolling(item.id)
    } else if (item.status === 'finished') {
      void refreshSamples(item.id)
    }
  }

  const onExport = async (values: ExportValues) => {
    if (!job) return
    setExporting(true)
    const yes = (v: string) => v === 'yes'
    try {
      const train = values.trainRatio
      const val = values.valRatio
      const test = Math.round((1 - train - val) * 100) / 100
      const res = await api.exportPrelabel(job.id, {
        out_name: values.out_name ?? '',
        task: values.task,
        name_style: values.name_style,
        file_mode: values.file_mode,
        overwrite: yes(values.overwrite),
        split: {
          enabled: yes(values.splitEnabled),
          ratios: [train, val, test],
          seed: values.seed,
          stratified: yes(values.stratified),
          respect_groups: yes(values.respectGroups),
          respect_existing: false,
        },
        clean: {
          enabled: yes(values.cleanEnabled),
          verify_readable: true,
          check_near_duplicates: yes(values.cleanNear),
          check_exact_duplicates: true,
        },
        taxonomy: { enabled: false },
      })
      setExportReport({ out_dir: res.out_dir, report: res.report })
      setJob(res.job)
      setStep(2)
      message.success('已生成数据集')
    } catch (err: any) {
      message.error(err.message ?? '生成数据集失败')
    } finally {
      setExporting(false)
    }
  }

  const report = job?.report ?? null
  const progressPercent = useMemo(() => {
    const p = job?.progress
    if (!p || !p.total) return 0
    return Math.min(100, Math.round((p.done / p.total) * 100))
  }, [job])

  /** 是/否 用的受控下拉（用字符串值，避免布尔值在 Select 里的取值歧义）。 */
  const yesNo = (yes = '是', no = '否') => [
    { value: 'yes', label: yes },
    { value: 'no', label: no },
  ]

  return (
    <div className="page">
      <Alert
        type="warning"
        showIcon
        icon={<WarningOutlined />}
        style={{ marginBottom: 16 }}
        message="预标注产出的是「伪标签」，不是标注"
        description={
          <span>
            模型漏检 = 这批数据<Text strong>少了标注</Text>；模型误检 = 把错误<Text strong>固化成标注</Text>。
            结果<Text strong>必须经人工复核后</Text>才能作为训练数据。血缘（权重 / conf / 时间）会写进
            <Text code>dataset_card.json</Text>，可追溯。
          </span>
        }
      />

      <Steps
        current={step}
        size="small"
        style={{ marginBottom: 16 }}
        items={[
          { title: '选择权重与图片', icon: <FolderOpenOutlined /> },
          { title: '预览与复核', icon: <ExperimentOutlined /> },
          { title: '生成数据集', icon: <CheckCircleOutlined /> },
        ]}
      />

      {step === 0 && (
        <Card title="步骤 1 · 权重 + 图片目录 → 批量预标注" size="small">
          <Form
            form={form}
            layout="vertical"
            onFinish={onStart}
            initialValues={{
              task: 'detect',
              device: 'cpu',
              imgsz: 640,
              conf: 0.25,
              iou: 0.7,
              maxDet: 300,
              limit: 0,
            }}
          >
            <Row gutter={16}>
              <Col xs={24} lg={12}>
                <Text type="secondary">权重</Text>
                <Select
                  style={{ width: '100%', marginTop: 4 }}
                  placeholder="从模型库 / 导出产物选择"
                  value={weightsValue || undefined}
                  onChange={setWeightsValue}
                  showSearch
                  optionFilterProp="label"
                  options={weights.map((w) => ({ value: w.value, label: w.label }))}
                  notFoundContent={<Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无可选项" />}
                />
                <Input
                  style={{ marginTop: 8, marginBottom: 16 }}
                  placeholder="或直接填写任意权重路径（.pt / .onnx / .torchscript）"
                  value={weightsValue}
                  onChange={(e) => setWeightsValue(e.target.value)}
                />
                {selectedFormat && (
                  <Alert
                    style={{ marginBottom: 16 }}
                    type={selectedFormat.available ? 'success' : 'warning'}
                    showIcon
                    message={`格式：${selectedFormat.label}`}
                    description={selectedFormat.available ? selectedFormat.task_hint : selectedFormat.reason}
                  />
                )}
              </Col>

              <Col xs={24} lg={12}>
                <Form.Item
                  name="images_dir"
                  label="待标注图片目录"
                  tooltip="会递归收集目录下的图片（jpg/png/bmp/webp/tif）"
                >
                  <Input prefix={<FolderOpenOutlined />} placeholder="例如 D:\data\to_label" allowClear />
                </Form.Item>
                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item name="task" label="任务类型">
                      <Select
                        options={[
                          { value: 'detect', label: '目标检测' },
                          { value: 'classify', label: '图像分类' },
                          { value: 'segment', label: '实例分割' },
                        ]}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item name="device" label="设备">
                      <Select
                        options={[
                          { value: 'cpu', label: 'CPU' },
                          { value: '0', label: 'GPU 0' },
                        ]}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col xs={24} lg={12}>
                <Form.Item
                  name="classesText"
                  label="类别清单（可选，逗号/分号分隔）"
                  tooltip="覆盖模型自带的类名；ONNX 未写入类名时建议填写"
                >
                  <Input placeholder="如：person,car,dog（留空用模型自带类名）" allowClear />
                </Form.Item>
              </Col>
              <Col xs={12} lg={6}>
                <Form.Item name="imgsz" label="推理尺寸">
                  <InputNumber style={{ width: '100%' }} min={32} max={1920} step={32} />
                </Form.Item>
              </Col>
              <Col xs={12} lg={6}>
                <Form.Item name="maxDet" label="每图最多框数">
                  <InputNumber style={{ width: '100%' }} min={1} max={1000} />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col xs={24} lg={8}>
                <Form.Item name="conf" label="置信度阈值 conf（低于它的框不会成为伪标签）">
                  <Slider min={0.01} max={1} step={0.01} />
                </Form.Item>
              </Col>
              <Col xs={24} lg={8}>
                <Form.Item name="iou" label="NMS IoU 阈值">
                  <Slider min={0.05} max={0.95} step={0.05} />
                </Form.Item>
              </Col>
              <Col xs={24} lg={8}>
                <Form.Item
                  name="limit"
                  label="只处理前 N 张（0 = 全部）"
                  tooltip="先跑少量图看看效果，再处理整批会更省时间"
                >
                  <InputNumber style={{ width: '100%' }} min={0} max={100000} />
                </Form.Item>
              </Col>
            </Row>

            <Space>
              <Button type="primary" htmlType="submit" loading={starting} icon={<TagsOutlined />}>
                开始预标注
              </Button>
              <Text type="secondary">CPU 上逐张推理，几百张图可能要几分钟；任务在后台跑，可离开本页。</Text>
            </Space>
          </Form>

          {jobs.length > 0 && (
            <Card
              type="inner"
              title="历史任务（仅本次后端进程内）"
              size="small"
              style={{ marginTop: 16 }}
              extra={
                <Button size="small" icon={<ReloadOutlined />} onClick={loadJobs}>
                  刷新
                </Button>
              }
            >
              <Table
                size="small"
                rowKey="id"
                pagination={false}
                dataSource={jobs}
                columns={[
                  { title: '任务', dataIndex: 'id', render: (v: string) => <Text className="mono">{v}</Text> },
                  { title: '图片目录', dataIndex: 'images_dir', ellipsis: true },
                  { title: '状态', width: 100, render: (_, r) => <StatusTag status={r.status} /> },
                  { title: '标注数', width: 90, render: (_, r) => r.report?.boxes_total ?? '-' },
                  {
                    title: '',
                    width: 80,
                    render: (_, r) => (
                      <Button size="small" type="link" onClick={() => watchJob(r)}>
                        查看
                      </Button>
                    ),
                  },
                ]}
              />
            </Card>
          )}
        </Card>
      )}

      {step === 1 && job && (
        <>
          <Card
            title={`步骤 2 · 预览与复核（${job.id}）`}
            size="small"
            extra={
              <Space>
                {job.status === 'finished' && (
                  <Button type="primary" onClick={() => setStep(2)} disabled={!job.ready}>
                    确认无误，生成数据集
                  </Button>
                )}
                <Button onClick={reset}>重新开始</Button>
              </Space>
            }
          >
            {job.status === 'failed' ? (
              <Alert type="error" showIcon message="预标注失败" description={job.error || '未知错误'} />
            ) : job.status === 'finished' ? (
              <>
                <Alert
                  type="success"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message="预标注完成"
                  description="下面是模型预测出的框。请注意：漏检的地方不会有框，误检的框也会被当成标注——请逐张复核。"
                />
                <Row gutter={[12, 12]}>
                  <Col xs={12} md={4}>
                    <Statistic title="图像总数" value={report?.images_total ?? 0} />
                  </Col>
                  <Col xs={12} md={4}>
                    <Statistic title="有标注" value={report?.images_with_boxes ?? 0} />
                  </Col>
                  <Col xs={12} md={4}>
                    <Statistic title="空标注" value={report?.images_empty ?? 0} />
                  </Col>
                  <Col xs={12} md={4}>
                    <Statistic
                      title="失败"
                      value={report?.images_failed ?? 0}
                      valueStyle={{ color: report?.images_failed ? '#cf1322' : undefined }}
                    />
                  </Col>
                  <Col xs={12} md={4}>
                    <Statistic
                      title={report?.task === 'classify' ? '图像标注' : '检测框'}
                      value={report?.boxes_total ?? 0}
                    />
                  </Col>
                  <Col xs={12} md={4}>
                    <Statistic title="耗时" value={report?.duration_sec ?? 0} suffix="s" />
                  </Col>
                </Row>
              </>
            ) : (
              <>
                <Paragraph style={{ marginBottom: 8 }}>
                  <Text type="secondary">
                    正在后台推理：{job.progress.done} / {job.progress.total} 张
                  </Text>
                </Paragraph>
                <Progress percent={progressPercent} status="active" />
                <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }} ellipsis>
                  {job.progress.current}
                </Paragraph>
              </>
            )}
          </Card>

          {report && (
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col xs={24} lg={12}>
                <Card title="逐类实例数" size="small">
                  <Table
                    size="small"
                    rowKey="name"
                    pagination={false}
                    scroll={{ y: 260 }}
                    dataSource={Object.entries(report.count_by_category)
                      .map(([name, count]) => ({ name, count }))
                      .sort((a, b) => b.count - a.count)}
                    columns={[
                      { title: '类别', dataIndex: 'name' },
                      { title: '实例数', dataIndex: 'count', width: 110, sorter: (a, b) => a.count - b.count },
                    ]}
                    locale={{ emptyText: '模型没有给出任何标注（可能阈值太高或图片中确实没有目标）' }}
                  />
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card title="血缘与参数" size="small">
                  <Descriptions column={1} size="small" bordered>
                    <Descriptions.Item label="权重">
                      <Text className="mono">{report.weights}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="任务">
                      {report.task}　<Text type="secondary">设备 {report.device}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="阈值">
                      conf {report.conf}　iou {report.iou}
                    </Descriptions.Item>
                    <Descriptions.Item label="模型类别">
                      {report.classes.length ? (
                        report.classes.map((c) => <Tag key={c}>{c}</Tag>)
                      ) : (
                        <Text type="secondary">（模型未提供类名）</Text>
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="生成时间">{report.created_at}</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>
          )}

          {report && report.failures.length > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginTop: 16 }}
              message={`${report.failures.length} 张图推理失败（已跳过，未中断整批）`}
              description={
                <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
                  {report.failures.slice(0, 8).map((f, i) => (
                    <li key={i}>
                      <Text className="mono">{f.path}</Text>：{f.error}
                    </li>
                  ))}
                  {report.failures.length > 8 && <li>…其余 {report.failures.length - 8} 条已省略</li>}
                </ul>
              }
            />
          )}

          {job.status === 'finished' && (
            <Card
              title={`预测框预览（前 ${samples.length} 张）`}
              size="small"
              style={{ marginTop: 16 }}
              extra={
                <Space>
                  {sampleCategories.map((c) => (
                    <Tag key={c}>{c}</Tag>
                  ))}
                </Space>
              }
            >
              {samples.length ? (
                <SampleGrid samples={samples} categories={sampleCategories} />
              ) : (
                <Empty description="没有可预览的样本" />
              )}
            </Card>
          )}
        </>
      )}

      {step === 2 && job && (
        <Card title="步骤 3 · 生成数据集" size="small" extra={<Button onClick={reset}>重新开始</Button>}>
          {exportReport ? (
            <>
              <Alert
                type="success"
                showIcon
                style={{ marginBottom: 12 }}
                message="数据集已生成"
                description={
                  <span>
                    输出目录：<Text className="mono">{exportReport.out_dir}</Text>
                    <br />
                    血缘已写入 <Text code>dataset_card.json</Text>（含 prelabel 段）。训练它之前，请先人工复核这批伪标签。
                  </span>
                }
              />
              <Descriptions column={2} size="small" bordered>
                <Descriptions.Item label="任务类型">{exportReport.report.task}</Descriptions.Item>
                <Descriptions.Item label="类别数">{exportReport.report.classes.length}</Descriptions.Item>
                <Descriptions.Item label="导出的检测框">{exportReport.report.boxes_exported}</Descriptions.Item>
                <Descriptions.Item label="跳过">{exportReport.report.images_skipped}</Descriptions.Item>
                <Descriptions.Item label="划分" span={2}>
                  {Object.entries(exportReport.report.images_exported).map(([k, v]) => (
                    <Tag key={k}>
                      {k}: {v}
                    </Tag>
                  ))}
                </Descriptions.Item>
              </Descriptions>
            </>
          ) : (
            <Form
              form={exportForm}
              layout="vertical"
              onFinish={onExport}
              initialValues={{
                out_name: `prelabel_${job.images_dir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? 'dataset'}`,
                task: job.task === 'classify' ? 'classification' : 'auto',
                name_style: 'keep',
                file_mode: 'copy',
                overwrite: 'no',
                splitEnabled: 'yes',
                trainRatio: 0.8,
                valRatio: 0.1,
                seed: 42,
                stratified: 'yes',
                respectGroups: 'yes',
                cleanEnabled: 'yes',
                cleanNear: 'no',
              }}
            >
              <Row gutter={16}>
                <Col xs={24} lg={8}>
                  <Form.Item
                    name="out_name"
                    label="输出目录名"
                    tooltip="会自动带上 prelabel 前缀，落在数据集存储目录下"
                  >
                    <Input />
                  </Form.Item>
                </Col>
                <Col xs={12} lg={4}>
                  <Form.Item name="task" label="导出任务">
                    <Select
                      options={[
                        { value: 'auto', label: '自动推断' },
                        { value: 'detection', label: '目标检测' },
                        { value: 'classification', label: '图像分类' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} lg={4}>
                  <Form.Item name="file_mode" label="文件方式">
                    <Select
                      options={[
                        { value: 'copy', label: '复制' },
                        { value: 'hardlink', label: '硬链接' },
                        { value: 'symlink', label: '软链接' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} lg={4}>
                  <Form.Item name="name_style" label="文件命名">
                    <Select
                      options={[
                        { value: 'keep', label: '保留原名' },
                        { value: 'source', label: '来源前缀' },
                        { value: 'uid', label: 'uid' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} lg={4}>
                  <Form.Item name="overwrite" label="覆盖已有">
                    <Select options={yesNo('是', '否')} />
                  </Form.Item>
                </Col>
              </Row>

              <Card type="inner" title="划分" size="small" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col xs={12} lg={4}>
                    <Form.Item name="splitEnabled" label="启用划分">
                      <Select options={yesNo()} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} lg={8}>
                    <Form.Item name="trainRatio" label="train 比例">
                      <Slider min={0.1} max={0.9} step={0.05} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} lg={8}>
                    <Form.Item name="valRatio" label="val 比例">
                      <Slider min={0} max={0.5} step={0.05} />
                    </Form.Item>
                  </Col>
                  <Col xs={12} lg={4}>
                    <Form.Item name="seed" label="随机种子">
                      <InputNumber style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
                <Space size="large">
                  <Form.Item name="stratified" label="按类别分层" style={{ marginBottom: 0 }}>
                    <Select options={yesNo()} style={{ width: 80 }} />
                  </Form.Item>
                  <Form.Item
                    name="respectGroups"
                    label="同组不拆散"
                    style={{ marginBottom: 0 }}
                    tooltip="视频抽帧的图片若被拆到不同子集会造成泄漏"
                  >
                    <Select options={yesNo()} style={{ width: 80 }} />
                  </Form.Item>
                </Space>
              </Card>

              <Card type="inner" title="清洗" size="small" style={{ marginBottom: 16 }}>
                <Space size="large">
                  <Form.Item name="cleanEnabled" label="导出前清洗" style={{ marginBottom: 0 }}>
                    <Select options={yesNo()} style={{ width: 80 }} />
                  </Form.Item>
                  <Form.Item name="cleanNear" label="近似重复检测（较慢）" style={{ marginBottom: 0 }}>
                    <Select
                      style={{ width: 100 }}
                      options={[
                        { value: 'no', label: '关闭' },
                        { value: 'yes', label: '开启' },
                      ]}
                    />
                  </Form.Item>
                </Space>
              </Card>

              <Space>
                <Button type="primary" htmlType="submit" loading={exporting}>
                  生成数据集
                </Button>
                <Button onClick={() => setStep(1)}>返回预览</Button>
              </Space>
            </Form>
          )}
        </Card>
      )}
    </div>
  )
}

function StatusTag({ status }: { status: PrelabelJob['status'] }) {
  const map: Record<PrelabelJob['status'], { color: string; text: string }> = {
    pending: { color: 'default', text: '排队中' },
    running: { color: 'processing', text: '运行中' },
    finished: { color: 'success', text: '已完成' },
    failed: { color: 'error', text: '失败' },
  }
  const item = map[status] ?? map.pending
  return <Tag color={item.color}>{item.text}</Tag>
}
