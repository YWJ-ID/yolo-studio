import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Empty,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons'
import { useNavigate, useSearchParams } from 'react-router-dom'
import ReactECharts from '../components/echarts'
import { api } from '../api/client'
import type { CompareResponse } from '../types'
import { formatMetric } from '../utils'

const { Text, Paragraph } = Typography

/** 多模型对比（M5-08 的对比部分）。不同划分的数字不可比，后端会给出提示。 */
export default function ModelCompare() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const ids = useMemo(
    () => (params.get('ids') ?? '').split(',').map((s) => s.trim()).filter(Boolean),
    [params],
  )

  const [split, setSplit] = useState('')
  const [data, setData] = useState<CompareResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    if (ids.length < 1) return
    setLoading(true)
    try {
      const res = await api.compareModels({ model_ids: ids, split: split || null })
      setData(res)
      setError('')
    } catch (err: any) {
      setError(err.message ?? '对比失败')
    } finally {
      setLoading(false)
    }
  }, [ids, split])

  useEffect(() => {
    load()
  }, [load])

  // 每个指标的最优模型标签（用于高亮），缺失指标不参与
  const bestByMetric = useMemo(() => {
    const out: Record<string, string> = {}
    if (!data) return out
    data.metrics.forEach((m) => {
      let bestLabel = ''
      let bestVal = -Infinity
      data.rows.forEach((row) => {
        const v = row.values[m]
        if (v != null && v > bestVal) {
          bestVal = v
          bestLabel = row.label
        }
      })
      if (bestLabel) out[m] = bestLabel
    })
    return out
  }, [data])

  const metricLabels = data?.metric_labels ?? data?.metrics ?? []

  if (ids.length < 2) {
    return (
      <div className="page">
        <Card size="small">
          <Empty description="请至少选择两个模型进行对比">
            <Button type="primary" onClick={() => navigate('/models')}>
              返回模型库
            </Button>
          </Empty>
        </Card>
      </div>
    )
  }

  const chartMetric = data?.metrics.find((m) => m === 'mAP50-95') ?? data?.metrics[0]

  return (
    <div className="page">
      <Card
        title={`模型对比（${ids.length} 个）`}
        size="small"
        extra={
          <Space>
            <Select
              size="small"
              style={{ width: 180 }}
              value={split}
              onChange={setSplit}
              options={[
                { value: '', label: '各模型最新一次评估' },
                { value: 'test', label: '只比较 test 划分' },
                { value: 'val', label: '只比较 val 划分' },
                { value: 'train', label: '只比较 train 划分' },
              ]}
            />
            <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
            <Button size="small" icon={<ArrowLeftOutlined />} onClick={() => navigate('/models')}>
              返回
            </Button>
          </Space>
        }
      >
        {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}
        {loading && <Spin style={{ display: 'block', margin: '24px auto' }} />}

        {data && !loading && (
          <>
            {(data.notes ?? []).map((n, i) => (
              <Alert
                key={i}
                type="warning"
                showIcon
                style={{ marginBottom: 8 }}
                message={n}
              />
            ))}

            {data.rows.length === 0 ? (
              <Empty description="没有可对比的评估结果（可能所选模型都还没评估）" />
            ) : (
              <>
                <Space wrap style={{ marginBottom: 8 }}>
                  <Text type="secondary">使用划分：</Text>
                  {data.splits.length ? (
                    data.splits.map((s) => (
                      <Tag key={s} color="geekblue">
                        {s}
                      </Tag>
                    ))
                  ) : (
                    <Tag>-</Tag>
                  )}
                </Space>

                <Table
                  size="small"
                  rowKey="label"
                  pagination={false}
                  scroll={{ x: 'max-content' }}
                  dataSource={data.rows}
                  columns={[
                    { title: '模型', dataIndex: 'label', fixed: 'left' as const },
                    { title: '划分', dataIndex: 'split', width: 80 },
                    { title: '类别数', dataIndex: 'num_classes', width: 80 },
                    ...data.metrics.map((m, idx) => ({
                      title: metricLabels[idx] ?? m,
                      key: m,
                      width: 120,
                      render: (_: unknown, row: (typeof data.rows)[number]) => {
                        const v = row.values[m]
                        if (v == null) return <Text type="secondary">-</Text>
                        const isBest = bestByMetric[m] === row.label
                        return (
                          <Text strong={isBest} style={{ color: isBest ? '#1677ff' : undefined }}>
                            {formatMetric(v)}
                            {isBest ? ' ★' : ''}
                          </Text>
                        )
                      },
                    })),
                  ]}
                />

                {chartMetric && (
                  <Card title={`${metricLabels[data.metrics.indexOf(chartMetric)] ?? chartMetric} 对比`} size="small" style={{ marginTop: 16 }}>
                    <ReactECharts
                      style={{ height: 260 }}
                      option={{
                        grid: { left: 8, right: 24, top: 16, bottom: 8, containLabel: true },
                        tooltip: { trigger: 'axis' },
                        xAxis: {
                          type: 'value',
                          axisLabel: { fontSize: 11, color: '#595959' },
                          splitLine: { lineStyle: { color: '#f5f5f5' } },
                        },
                        yAxis: {
                          type: 'category',
                          inverse: true,
                          data: data.rows.map((r) => r.label),
                          axisLabel: { fontSize: 11, color: '#595959' },
                        },
                        series: [
                          {
                            type: 'bar',
                            barWidth: '46%',
                            data: data.rows.map((r) => r.values[chartMetric] ?? 0),
                            itemStyle: { color: '#1677ff', borderRadius: [0, 3, 3, 0] },
                            label: {
                              show: true,
                              position: 'right',
                              fontSize: 11,
                              color: '#595959',
                              formatter: (p: any) => formatMetric(p.value),
                            },
                          },
                        ],
                      }}
                    />
                  </Card>
                )}

                {data.class_table.length > 0 && (
                  <Card title="逐类 AP50-95 对比（缺失留空，不补 0）" size="small" style={{ marginTop: 16 }}>
                    <Table
                      size="small"
                      rowKey="name"
                      pagination={false}
                      scroll={{ x: 'max-content' }}
                      dataSource={data.class_table}
                      columns={[
                        { title: '类别', dataIndex: 'name', fixed: 'left' as const },
                        ...data.rows.map((row) => ({
                          title: row.label,
                          key: row.label,
                          width: 130,
                          render: (_: unknown, r: { name: string; values: Record<string, number | null> }) => {
                            const v = r.values[row.label]
                            if (v == null) return <Text type="secondary">-</Text>
                            const columnValues = data.class_table
                              .map((ct) => ct.values[row.label])
                              .filter((x): x is number => x != null)
                            const isBest = columnValues.length > 0 && v === Math.max(...columnValues)
                            return (
                              <Text strong={isBest} style={{ color: isBest ? '#1677ff' : undefined }}>
                                {formatMetric(v)}
                              </Text>
                            )
                          },
                        })),
                      ]}
                    />
                    <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                      高亮为该列最优；空白表示该模型没有这个类别（或没有该类的指标）。
                    </Paragraph>
                  </Card>
                )}
              </>
            )}
          </>
        )}
      </Card>
    </div>
  )
}
