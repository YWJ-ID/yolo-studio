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
  InputNumber,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { BarChartOutlined, FileTextOutlined } from '@ant-design/icons'
import ReactECharts from '../components/echarts'
import { api } from '../api/client'
import type { AnalyticsReport } from '../types'
import { SampleGrid } from '../components/SampleGrid'

const { Text, Paragraph } = Typography

const LEVEL_STYLE: Record<string, { type: 'error' | 'warning' | 'success'; label: string }> = {
  error: { type: 'error', label: '严重' },
  warning: { type: 'warning', label: '警告' },
  info: { type: 'success', label: '提示' },
}

const AXIS_STYLE = {
  axisLabel: { fontSize: 11, color: '#595959' },
  axisLine: { lineStyle: { color: '#e8e8e8' } },
  splitLine: { lineStyle: { color: '#f5f5f5' } },
}

export default function QualityReport() {
  const [form] = Form.useForm()
  const [report, setReport] = useState<AnalyticsReport | null>(null)
  const [busy, setBusy] = useState<'' | 'analyze' | 'export'>('')
  const [reportUrl, setReportUrl] = useState('')

  const buildPayload = () => {
    const v = form.getFieldsValue()
    return {
      path: String(v.path || '').trim(),
      group_by: v.group_by,
      analytics: {
        sample_size: v.sample_size ?? 24,
        min_class_instances: v.min_class ?? 20,
        max_image_sizes: 12,
      },
    }
  }

  const onAnalyze = async () => {
    const payload = buildPayload()
    if (!payload.path) {
      message.warning('请输入数据目录')
      return
    }
    setBusy('analyze')
    setReportUrl('')
    try {
      const res = await api.analytics(payload)
      setReport(res.report)
      message.success('分析完成')
    } catch (err: any) {
      message.error(err.message ?? '分析失败')
    } finally {
      setBusy('')
    }
  }

  const onExportHtml = async () => {
    const payload = buildPayload()
    setBusy('export')
    try {
      const res = await api.analyticsReport({ ...payload, title: `质量报告 · ${payload.path}`, embed_samples: true })
      setReportUrl(res.url)
      message.success(`报告已生成（${(res.size_bytes / 1024).toFixed(1)} KB）`)
    } catch (err: any) {
      message.error(err.message ?? '生成报告失败')
    } finally {
      setBusy('')
    }
  }

  const s = report?.summary

  return (
    <div className="page">
      <Card title="数据集质量报告" size="small">
        <Form
          form={form}
          layout="inline"
          initialValues={{ sample_size: 24, min_class: 20, group_by: 'none' }}
          style={{ rowGap: 12 }}
        >
          <Form.Item name="path" label="数据目录" style={{ flex: 1, minWidth: 380 }}>
            <Input placeholder="例如 D:\dataset\archive" allowClear />
          </Form.Item>
          <Form.Item name="sample_size" label="抽样张数">
            <InputNumber min={0} max={200} style={{ width: 90 }} />
          </Form.Item>
          <Form.Item name="min_class" label="类别最少实例" tooltip="低于此值会提示样本不足">
            <InputNumber min={0} style={{ width: 110 }} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" icon={<BarChartOutlined />} loading={busy === 'analyze'} onClick={onAnalyze}>
                分析
              </Button>
              <Button
                icon={<FileTextOutlined />}
                loading={busy === 'export'}
                onClick={onExportHtml}
                disabled={!report}
              >
                导出 HTML 报告
              </Button>
            </Space>
          </Form.Item>
        </Form>

        {reportUrl && (
          <Alert
            style={{ marginTop: 12 }}
            type="success"
            showIcon
            message="HTML 报告已生成（单文件、离线可看、已内嵌缩略图）"
            description={
              <a href={reportUrl} target="_blank" rel="noreferrer">
                点击下载
              </a>
            }
          />
        )}
      </Card>

      {busy === 'analyze' && (
        <Card size="small" style={{ marginTop: 16 }}>
          <Spin tip="正在统计（大图集需要一些时间）…">
            <div style={{ height: 80 }} />
          </Spin>
        </Card>
      )}

      {report && s && (
        <>
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic title="图像" value={s.num_images} />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic title="标注" value={s.num_annotations} />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic title="类别" value={s.num_classes} />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic title="分组" value={s.num_groups} />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic title="标注形态" value={s.annotation_kind} valueStyle={{ fontSize: 16 }} />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic
                  title="未划分"
                  value={s.unassigned_images}
                  valueStyle={{ color: s.unassigned_images ? '#fa8c16' : undefined }}
                />
              </Card>
            </Col>
          </Row>

          {report.findings.length > 0 && (
            <Card title="检查结论" size="small" style={{ marginTop: 16 }}>
              {report.findings.map((f, i) => {
                const style = LEVEL_STYLE[f.level] ?? LEVEL_STYLE.info
                return (
                  <Alert
                    key={i}
                    type={style.type}
                    showIcon
                    style={{ marginBottom: 8 }}
                    message={`[${style.label}] ${f.message}`}
                  />
                )
              })}
            </Card>
          )}

          {report.recommendations.length > 0 && (
            <Card title="处置建议" size="small" style={{ marginTop: 16 }}>
              <ul style={{ marginBottom: 0, paddingLeft: 20 }}>
                {report.recommendations.map((r, i) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    {r}
                  </li>
                ))}
              </ul>
            </Card>
          )}

          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={24} lg={12}>
              <Card
                title="类别分布"
                size="small"
                extra={
                  report.imbalance.ratio ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      最大/最小 = {report.imbalance.ratio}×
                    </Text>
                  ) : null
                }
              >
                <ReactECharts
                  style={{ height: 300 }}
                  option={{
                    grid: { left: 8, right: 40, top: 12, bottom: 8, containLabel: true },
                    tooltip: { trigger: 'axis' },
                    xAxis: { type: 'value', ...AXIS_STYLE },
                    yAxis: {
                      type: 'category',
                      inverse: true,
                      data: report.class_distribution.map((c) => c.name),
                      ...AXIS_STYLE,
                    },
                    series: [
                      {
                        type: 'bar',
                        data: report.class_distribution.map((c) => c.count),
                        itemStyle: { color: '#1677ff', borderRadius: [0, 3, 3, 0] },
                        label: { show: true, position: 'right', fontSize: 11, color: '#595959' },
                      },
                    ],
                  }}
                />
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="目标尺寸分布" size="small">
                {s.num_bbox_annotations > 0 ? (
                  <ReactECharts
                    style={{ height: 300 }}
                    option={{
                      grid: { left: 8, right: 16, top: 24, bottom: 8, containLabel: true },
                      tooltip: { trigger: 'axis' },
                      xAxis: {
                        type: 'category',
                        data: report.size_category.map((x) => x.name),
                        ...AXIS_STYLE,
                      },
                      yAxis: { type: 'value', ...AXIS_STYLE },
                      series: [
                        {
                          type: 'bar',
                          barWidth: '42%',
                          data: report.size_category.map((x) => x.count),
                          itemStyle: {
                            color: (p: any) =>
                              ['#52c41a', '#1677ff', '#722ed1'][p.dataIndex] ?? '#1677ff',
                            borderRadius: [3, 3, 0, 0],
                          },
                          label: { show: true, position: 'top', fontSize: 11, color: '#595959' },
                        },
                      ],
                    }}
                  />
                ) : (
                  <Empty description="当前数据没有检测框（图像级标注无尺寸概念）" />
                )}
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="标注框面积占比" size="small">
                <ReactECharts
                  style={{ height: 260 }}
                  option={{
                    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
                    tooltip: { trigger: 'axis' },
                    xAxis: {
                      type: 'category',
                      data: report.area_ratio_histogram.labels,
                      axisLabel: { fontSize: 10, color: '#595959', rotate: 30 },
                      axisLine: { lineStyle: { color: '#e8e8e8' } },
                    },
                    yAxis: { type: 'value', ...AXIS_STYLE },
                    series: [
                      {
                        type: 'bar',
                        data: report.area_ratio_histogram.counts,
                        itemStyle: { color: '#13c2c2', borderRadius: [3, 3, 0, 0] },
                      },
                    ],
                  }}
                />
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="每图目标数" size="small">
                <ReactECharts
                  style={{ height: 260 }}
                  option={{
                    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
                    tooltip: { trigger: 'axis' },
                    xAxis: {
                      type: 'category',
                      data: report.objects_per_image_histogram.labels,
                      axisLabel: { fontSize: 10, color: '#595959', rotate: 30 },
                      axisLine: { lineStyle: { color: '#e8e8e8' } },
                    },
                    yAxis: { type: 'value', ...AXIS_STYLE },
                    series: [
                      {
                        type: 'bar',
                        data: report.objects_per_image_histogram.counts,
                        itemStyle: { color: '#faad14', borderRadius: [3, 3, 0, 0] },
                      },
                    ],
                  }}
                />
              </Card>
            </Col>
          </Row>

          <Card title="类别 × 子集" size="small" style={{ marginTop: 16 }}>
            <Table
              size="small"
              rowKey="cls"
              pagination={false}
              scroll={{ x: 'max-content' }}
              dataSource={report.class_distribution.map((c) => ({ cls: c.name }))}
              columns={[
                { title: '类别', dataIndex: 'cls', fixed: 'left' as const },
                ...['train', 'val', 'test'].map((sp) => ({
                  title: sp,
                  width: 110,
                  render: (_: unknown, r: { cls: string }) => {
                    const v = report.class_by_split?.[sp]?.[r.cls] ?? 0
                    return v === 0 ? <Text type="danger">0</Text> : v
                  },
                })),
                {
                  title: '合计',
                  width: 100,
                  render: (_: unknown, r: { cls: string }) =>
                    report.class_distribution.find((c) => c.name === r.cls)?.count ?? 0,
                },
              ]}
            />
            <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
              红色 0 表示该类别在此子集中完全缺失，对应指标会是 NaN。
            </Paragraph>
          </Card>

          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={24} lg={10}>
              <Card title="图像尺寸" size="small">
                <Table
                  size="small"
                  rowKey="size"
                  pagination={false}
                  dataSource={report.image_sizes}
                  columns={[
                    { title: '尺寸', dataIndex: 'size' },
                    { title: '数量', dataIndex: 'count', width: 100 },
                  ]}
                />
              </Card>
            </Col>
            <Col xs={24} lg={14}>
              <Card title="子集概览" size="small">
                <Table
                  size="small"
                  rowKey="split"
                  pagination={false}
                  dataSource={report.split_summary}
                  columns={[
                    { title: '子集', dataIndex: 'split', width: 90 },
                    { title: '图像', dataIndex: 'images', width: 90 },
                    { title: '标注', dataIndex: 'annotations', width: 90 },
                    { title: '检测框', dataIndex: 'bbox_annotations', width: 90 },
                    {
                      title: '覆盖类别数',
                      dataIndex: 'classes_present',
                      render: (v: number) => (
                        <Tag color={v < s.num_classes ? 'orange' : 'green'}>
                          {v}/{s.num_classes}
                        </Tag>
                      ),
                    },
                  ]}
                />
              </Card>
            </Col>
          </Row>

          {report.samples.length > 0 && (
            <Card
              title={`抽样预览（${report.samples.length} 张，按类别轮转抽取）`}
              size="small"
              style={{ marginTop: 16 }}
            >
              <SampleGrid
                samples={report.samples}
                categories={report.class_distribution.map((c) => c.name)}
              />
            </Card>
          )}
        </>
      )}

      {!report && busy !== 'analyze' && (
        <Card size="small" style={{ marginTop: 16 }}>
          <Empty description="输入数据目录后点击「分析」，查看分布统计与质量结论" />
        </Card>
      )}
    </div>
  )
}
