import { useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { FolderOpenOutlined, ReloadOutlined } from '@ant-design/icons'
import { api } from '../api/client'
import type { ScanResponse } from '../types'
import { SampleGrid } from '../components/SampleGrid'
import ExportStep, { SourceSpec } from '../components/ExportStep'

const { Text, Paragraph } = Typography

interface FormValues {
  path: string
  fmt?: string
  group_by?: string
  level?: string
  frames_dir?: string
  include_objects?: boolean
  images_dir?: string
  voc_one_based?: boolean
}

/** 合成清单里的一项：来源参数 + 扫描到的形态与类别。 */
interface SourceEntry {
  spec: SourceSpec
  kind: string
  categories: string[]
}

export default function DatasetImport() {
  const [form] = Form.useForm<FormValues>()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScanResponse | null>(null)
  const [pending, setPending] = useState<SourceEntry | null>(null)
  const [sources, setSources] = useState<SourceEntry[]>([])

  const onScan = async (values: FormValues) => {
    if (!values.path?.trim()) {
      message.warning('请输入数据目录的绝对路径')
      return
    }
    setLoading(true)
    setResult(null)
    setPending(null)
    const spec: SourceSpec = {
      path: values.path.trim(),
      fmt: values.fmt,
      group_by: values.group_by,
      level: values.level,
      frames_dir: values.frames_dir,
      include_objects: values.include_objects,
      images_dir: values.images_dir,
      voc_one_based: values.voc_one_based,
    }
    try {
      const res = await api.scan(spec)
      setResult(res)
      setPending({ spec, kind: res.stats.annotation_kind, categories: res.categories })
      message.success(`识别为 ${res.detected_format.toUpperCase()} 格式，共 ${res.stats.num_images} 张图`)
    } catch (err: any) {
      message.error(err.message ?? '扫描失败')
    } finally {
      setLoading(false)
    }
  }

  const addToSources = () => {
    if (!pending) return
    if (sources.some((s) => s.spec.path === pending.spec.path)) {
      message.warning('该来源已在合成清单中')
      return
    }
    setSources([...sources, pending])
    message.success('已加入合成清单，可继续扫描其它来源')
  }

  // 合并后的标注形态与类别并集（客户端估算，用于步骤 2 的提示与类别顺序）
  const mergedKind = (() => {
    const kinds = new Set(sources.map((s) => s.kind))
    if (kinds.size === 0) return 'unknown'
    if (kinds.size > 1) return 'mixed'
    return [...kinds][0]
  })()
  const mergedCategories: string[] = []
  sources.forEach((s) => s.categories.forEach((c) => !mergedCategories.includes(c) && mergedCategories.push(c)))

  const stats = result?.stats

  return (
    <div className="page">
      <Card title="数据导入 · 步骤 1：扫描与预览" size="small">
        <Form
          form={form}
          layout="inline"
          onFinish={onScan}
          initialValues={{ group_by: 'none' }}
          style={{ rowGap: 12 }}
        >
          <Form.Item
            name="path"
            label="数据目录"
            style={{ flex: 1, minWidth: 420 }}
            tooltip="支持 Roboflow / ultralytics 结构的 YOLO 数据集根目录"
          >
            <Input
              prefix={<FolderOpenOutlined />}
              placeholder="例如 D:\dataset\archive"
              allowClear
            />
          </Form.Item>

          <Form.Item name="fmt" label="格式">
            <Select
              style={{ width: 130 }}
              allowClear
              placeholder="自动探测"
              options={[
                { value: 'yolo', label: 'YOLO' },
                { value: 'coco', label: 'COCO' },
                { value: 'voc', label: 'VOC' },
                { value: 'labelme', label: 'LabelMe' },
                { value: 'openlabel', label: 'OpenLABEL' },
              ]}
            />
          </Form.Item>

          <Form.Item name="group_by" label="分组键" tooltip="防泄漏划分：同一视频的帧应整体落在同一子集">
            <Select
              style={{ width: 120 }}
              options={[
                { value: 'none', label: '不分组' },
                { value: 'parent', label: '按子目录' },
                { value: 'stem', label: '按文件名前缀' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="level"
            label="标注层级"
            tooltip="仅 OpenLABEL/VCD 使用。留空走默认 driver_actions；填 - 表示使用全部层级"
          >
            <Input style={{ width: 160 }} placeholder="driver_actions" allowClear />
          </Form.Item>

          <Form.Item name="frames_dir" label="帧目录" tooltip="仅 OpenLABEL/VCD 使用，留空自动探测">
            <Input style={{ width: 130 }} placeholder="自动" allowClear />
          </Form.Item>

          <Form.Item name="include_objects" label="并入object" valuePropName="checked">
            <Switch size="small" />
          </Form.Item>

          <Form.Item name="images_dir" label="图像目录" tooltip="仅 COCO/VOC 使用，留空自动探测">
            <Input style={{ width: 130 }} placeholder="自动" allowClear />
          </Form.Item>

          <Form.Item
            name="voc_one_based"
            label="VOC 1-based"
            valuePropName="checked"
            tooltip="VOC 原始规范坐标是 1-based 闭区间；LabelImg 产出的是 0-based。默认按 0-based 处理"
          >
            <Switch size="small" />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} icon={<ReloadOutlined />}>
              扫描
            </Button>
          </Form.Item>
        </Form>

        <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
          本步骤只做「读取 + 统计」，<Text strong>不会修改任何原始文件</Text>。清洗与合并将在后续步骤进行。
          <br />
          已支持 YOLO / COCO / VOC / LabelMe / OpenLABEL 五种格式，格式自动探测。
          OpenLABEL/VCD（如 DMD 数据集）是<Text strong>逐帧动作标注</Text>，没有边界框，
          对应 YOLO 的<Text strong>分类</Text>任务。
        </Paragraph>
      </Card>

      {loading && (
        <Card size="small" style={{ marginTop: 16 }}>
          <Spin tip="正在扫描并解析标签…">
            <div style={{ height: 80 }} />
          </Spin>
        </Card>
      )}

      {result && stats && (
        <>
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={12} md={6}>
              <Card size="small">
                <Statistic title="图像总数" value={stats.num_images} />
              </Card>
            </Col>
            <Col xs={12} md={6}>
              <Card size="small">
                <Statistic title="标注总数" value={stats.num_annotations} />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  检测框 {stats.num_bbox_annotations} / 图像级 {stats.num_image_labels}
                </Text>
              </Card>
            </Col>
            <Col xs={12} md={6}>
              <Card size="small">
                <Statistic title="类别数" value={stats.num_categories} />
              </Card>
            </Col>
            <Col xs={12} md={6}>
              <Card size="small">
                <div className="ant-statistic-title">标注形态</div>
                <div style={{ marginTop: 8 }}>
                  {stats.annotation_kind === 'bbox' && <Tag color="blue">目标检测（bbox）</Tag>}
                  {stats.annotation_kind === 'image' && <Tag color="purple">图像分类（无框）</Tag>}
                  {stats.annotation_kind === 'mixed' && <Tag color="orange">混合</Tag>}
                  {stats.annotation_kind === 'unknown' && <Tag>无标注</Tag>}
                </div>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={24} lg={12}>
              <Card title="划分分布" size="small">
                <Descriptions column={1} size="small">
                  {Object.entries(stats.split_counts).map(([k, v]) => (
                    <Descriptions.Item key={k} label={k}>
                      {v} 张
                    </Descriptions.Item>
                  ))}
                  {stats.unassigned_images > 0 && (
                    <Descriptions.Item label="未划分">{stats.unassigned_images} 张</Descriptions.Item>
                  )}
                  {stats.num_groups > 0 && (
                    <Descriptions.Item label="分组数">
                      {stats.num_groups}
                      <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                        划分时同组不可拆散
                      </Text>
                    </Descriptions.Item>
                  )}
                </Descriptions>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="类别分布" size="small">
                <Table
                  size="small"
                  rowKey="name"
                  pagination={false}
                  scroll={{ y: 240 }}
                  dataSource={Object.entries(stats.count_by_category)
                    .map(([name, count]) => ({ name, count }))
                    .sort((a, b) => b.count - a.count)}
                  columns={[
                    { title: '类别', dataIndex: 'name' },
                    { title: '实例数', dataIndex: 'count', width: 110, sorter: (a, b) => a.count - b.count },
                    {
                      title: '占比',
                      width: 90,
                      render: (_, r) =>
                        stats.num_annotations
                          ? `${((r.count / stats.num_annotations) * 100).toFixed(1)}%`
                          : '-',
                    },
                  ]}
                />
              </Card>
            </Col>
          </Row>

          {result.warnings.length > 0 && (
            <Alert
              style={{ marginTop: 16 }}
              type="warning"
              showIcon
              message={`发现 ${result.warnings.length} 条接入警告（完整清洗见任务 M1-08）`}
              description={
                <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
                  {result.warnings.slice(0, 8).map((w, i) => (
                    <li key={i}>
                      <Text type="secondary">{w}</Text>
                    </li>
                  ))}
                  {result.warnings.length > 8 && <li>…其余 {result.warnings.length - 8} 条已省略</li>}
                </ul>
              }
            />
          )}

          <Card
            title={`样本预览（前 ${result.samples.length} 张）`}
            size="small"
            style={{ marginTop: 16 }}
            extra={
              <Space>
                {result.categories.map((c) => (
                  <Tag key={c}>{c}</Tag>
                ))}
              </Space>
            }
          >
            {result.samples.length ? (
              <SampleGrid samples={result.samples} categories={result.categories} />
            ) : (
              <Empty description="没有解析到图像" />
            )}
          </Card>

          <Card size="small" style={{ marginTop: 16 }}>
            <Space>
              <Button type="primary" onClick={addToSources} disabled={!pending}>
                加入合成清单
              </Button>
              <Text type="secondary">
                把当前来源加入待合并列表，可继续扫描其它来源，实现多格式数据合成同一个数据集
              </Text>
            </Space>
          </Card>
        </>
      )}

      {sources.length > 0 && (
        <Card
          title={`合成清单（${sources.length} 个来源）`}
          size="small"
          style={{ marginTop: 16 }}
          extra={
            <Button size="small" danger onClick={() => setSources([])}>
              清空
            </Button>
          }
        >
          <Table
            size="small"
            rowKey={(r) => r.spec.path}
            pagination={false}
            dataSource={sources}
            columns={[
              { title: '路径', dataIndex: ['spec', 'path'], render: (v: string) => <Text className="mono">{v}</Text> },
              {
                title: '形态',
                width: 110,
                render: (_, r) =>
                  r.kind === 'bbox' ? (
                    <Tag color="blue">检测框</Tag>
                  ) : r.kind === 'image' ? (
                    <Tag color="purple">图像级</Tag>
                  ) : (
                    <Tag>{r.kind}</Tag>
                  ),
              },
              { title: '类别数', width: 80, render: (_, r) => r.categories.length },
              {
                title: '',
                width: 70,
                render: (_, r) => (
                  <Button
                    size="small"
                    type="link"
                    danger
                    onClick={() => setSources(sources.filter((s) => s.spec.path !== r.spec.path))}
                  >
                    移除
                  </Button>
                ),
              },
            ]}
          />
          {mergedKind === 'mixed' && (
            <Alert
              style={{ marginTop: 12 }}
              type="warning"
              showIcon
              message="清单中存在不同标注形态的来源"
              description="合并后会得到混合数据集，导出时需显式指定任务类型；单一目录布局无法同时表达检测与分类，建议先只合并同形态的来源。"
            />
          )}
        </Card>
      )}

      {sources.length > 0 && (
        <ExportStep
          sources={sources.map((s) => s.spec)}
          annotationKind={mergedKind}
          categories={mergedCategories}
        />
      )}

      {!result && !loading && (
        <Card size="small" style={{ marginTop: 16 }}>
          <Empty description="输入数据目录后点击「扫描」查看统计与样本预览" />
        </Card>
      )}
    </div>
  )
}
