import { useEffect, useState } from 'react'
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Col,
  Empty,
  Form,
  Input,
  InputNumber,
  Pagination,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Tag,
  Typography,
  message,
} from 'antd'
import { PictureOutlined, ReloadOutlined } from '@ant-design/icons'
import { api } from '../api/client'
import type { BrowseResponse, DatasetVersion } from '../types'
import { SampleGrid } from '../components/SampleGrid'

const { Text, Paragraph } = Typography

const PAGE_SIZE = 24

interface Filters {
  path: string
  fmt?: string
  category?: string
  split?: string
  kind?: string
  source_id?: string
  search?: string
  min_objects?: number
  max_objects?: number
  min_edge?: number
  max_edge?: number
}

/** 数据浏览器（M5-04）：缩略图 + 标注框叠加，按类别 / 来源 / 划分 / 尺寸筛选。 */
export default function DatasetBrowser() {
  const [form] = Form.useForm()
  const [versions, setVersions] = useState<DatasetVersion[]>([])
  const [applied, setApplied] = useState<Filters | null>(null)
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<BrowseResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.versions().then(setVersions).catch(() => undefined)
  }, [])

  const run = (nextOffset = 0) => {
    const v = form.getFieldsValue()
    if (!v.path?.trim()) {
      message.warning('请选择或输入数据目录')
      return
    }
    setOffset(nextOffset)
    setApplied({
      path: v.path.trim(),
      fmt: v.fmt,
      category: v.category,
      split: v.split,
      kind: v.kind,
      source_id: v.source_id,
      search: v.search?.trim() || undefined,
      min_objects: v.min_objects ?? undefined,
      max_objects: v.max_objects ?? undefined,
      min_edge: v.min_edge ?? undefined,
      max_edge: v.max_edge ?? undefined,
    })
  }

  useEffect(() => {
    if (!applied) return
    setLoading(true)
    api
      .browse({
        path: applied.path,
        fmt: applied.fmt,
        browse: {
          offset,
          limit: PAGE_SIZE,
          category: applied.category ?? null,
          split: applied.split ?? null,
          kind: applied.kind ?? null,
          source_id: applied.source_id ?? null,
          search: applied.search ?? null,
          min_objects: applied.min_objects ?? null,
          max_objects: applied.max_objects ?? null,
          min_edge: applied.min_edge ?? null,
          max_edge: applied.max_edge ?? null,
        },
      })
      .then((res) => {
        setData(res)
        setError('')
      })
      .catch((err) => setError(err.message ?? '浏览失败'))
      .finally(() => setLoading(false))
  }, [applied, offset])

  const pathOptions = versions.map((v) => ({
    value: v.path,
    label: `${v.name}（${v.task}）`,
  }))

  const stats = data?.stats

  return (
    <div className="page">
      <Card title="数据浏览器 · 筛选" size="small">
        <Form
          form={form}
          layout="inline"
          style={{ rowGap: 12 }}
          onFinish={() => run(0)}
          initialValues={{ fmt: undefined, kind: undefined }}
        >
          <Form.Item name="path" label="数据目录" style={{ flex: 1, minWidth: 360 }}>
            <AutoComplete
              options={pathOptions}
              placeholder="选择已生成的数据集，或输入任意数据集目录"
              filterOption={(input, option) =>
                String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())
              }
            />
          </Form.Item>

          <Form.Item name="fmt" label="格式">
            <Select
              style={{ width: 120 }}
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

          <Form.Item name="split" label="划分">
            <Select
              style={{ width: 100 }}
              allowClear
              placeholder="全部"
              options={['train', 'val', 'test'].map((s) => ({ value: s, label: s }))}
            />
          </Form.Item>

          <Form.Item name="kind" label="形态">
            <Select
              style={{ width: 120 }}
              allowClear
              placeholder="全部"
              options={[
                { value: 'bbox', label: '有检测框' },
                { value: 'image', label: '图像级' },
              ]}
            />
          </Form.Item>

          <Form.Item name="category" label="类别">
            <Select
              style={{ width: 140 }}
              allowClear
              showSearch
              placeholder="全部"
              options={(data?.categories ?? []).map((c) => ({ value: c, label: c }))}
            />
          </Form.Item>

          <Form.Item name="source_id" label="来源">
            <Select
              style={{ width: 160 }}
              allowClear
              placeholder="全部"
              options={(data?.sources ?? []).map((s) => ({
                value: s.source_id,
                label: `${s.source_id}（${s.images}）`,
              }))}
            />
          </Form.Item>

          <Form.Item name="search" label="路径包含">
            <Input style={{ width: 150 }} placeholder="子串" allowClear />
          </Form.Item>

          <Form.Item name="min_objects" label="目标数≥">
            <InputNumber min={0} style={{ width: 80 }} />
          </Form.Item>
          <Form.Item name="max_objects" label="目标数≤">
            <InputNumber min={0} style={{ width: 80 }} />
          </Form.Item>

          <Form.Item name="min_edge" label="长边≥" tooltip="图像长边 = max(宽, 高)，单位像素">
            <InputNumber min={0} style={{ width: 90 }} />
          </Form.Item>
          <Form.Item name="max_edge" label="长边≤">
            <InputNumber min={0} style={{ width: 90 }} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<PictureOutlined />} loading={loading}>
                浏览
              </Button>
              {applied && (
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => run(offset)}
                  disabled={loading}
                >
                  刷新本页
                </Button>
              )}
            </Space>
          </Form.Item>
        </Form>

        <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
          只读浏览：缩略图按 <Text strong>原图坐标</Text> 叠加标注框，同一类别在任意图上颜色一致。
          检测数据画框，分类数据（图像级标注）只显示类别标签。
        </Paragraph>
      </Card>

      {error && <Alert type="error" showIcon message={error} style={{ marginTop: 16 }} />}

      {loading && (
        <Card size="small" style={{ marginTop: 16 }}>
          <Spin tip="正在解析并筛选…">
            <div style={{ height: 60 }} />
          </Spin>
        </Card>
      )}

      {data && stats && !loading && (
        <>
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={12} md={5}>
              <Card size="small">
                <Statistic title="匹配图像" value={data.total} />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  数据集共 {stats.num_images} 张
                </Text>
              </Card>
            </Col>
            <Col xs={12} md={5}>
              <Card size="small">
                <Statistic title="标注总数" value={stats.num_annotations} />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  检测框 {stats.num_bbox_annotations} / 图像级 {stats.num_image_labels}
                </Text>
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic title="类别数" value={stats.num_categories} />
              </Card>
            </Col>
            <Col xs={12} md={5}>
              <Card size="small">
                <div className="ant-statistic-title">标注形态</div>
                <div style={{ marginTop: 8 }}>
                  {stats.annotation_kind === 'bbox' && <Tag color="blue">目标检测</Tag>}
                  {stats.annotation_kind === 'image' && <Tag color="purple">图像分类</Tag>}
                  {stats.annotation_kind === 'mixed' && <Tag color="orange">混合</Tag>}
                  {stats.annotation_kind === 'unknown' && <Tag>无标注</Tag>}
                </div>
              </Card>
            </Col>
            <Col xs={12} md={5}>
              <Card size="small">
                <div className="ant-statistic-title">来源</div>
                <div style={{ marginTop: 8 }}>
                  <Space wrap size={4}>
                    {data.sources.map((s) => (
                      <Tag key={s.source_id} color="geekblue">
                        {s.format} · {s.images}
                      </Tag>
                    ))}
                  </Space>
                </div>
              </Card>
            </Col>
          </Row>

          <Card
            title={`样本预览（第 ${Math.floor(offset / PAGE_SIZE) + 1} 页）`}
            size="small"
            style={{ marginTop: 16 }}
            extra={
              data.categories.length > 0 && (
                <Space wrap size={4} style={{ maxWidth: 700, justifyContent: 'flex-end' }}>
                  {data.categories.slice(0, 12).map((c) => (
                    <Tag key={c}>{c}</Tag>
                  ))}
                  {data.categories.length > 12 && <Tag>…共 {data.categories.length} 类</Tag>}
                </Space>
              )
            }
          >
            {data.images.length ? (
              <SampleGrid samples={data.images} categories={data.categories} />
            ) : (
              <Empty description="没有符合筛选条件的图像" />
            )}

            {data.total > PAGE_SIZE && (
              <div style={{ marginTop: 16, textAlign: 'right' }}>
                <Pagination
                  current={Math.floor(offset / PAGE_SIZE) + 1}
                  pageSize={PAGE_SIZE}
                  total={data.total}
                  showSizeChanger={false}
                  onChange={(page) => run((page - 1) * PAGE_SIZE)}
                />
              </div>
            )}
          </Card>
        </>
      )}

      {!data && !loading && (
        <Card size="small" style={{ marginTop: 16 }}>
          <Empty description="选择数据目录后点击「浏览」，查看缩略图与标注叠加" />
        </Card>
      )}
    </div>
  )
}
