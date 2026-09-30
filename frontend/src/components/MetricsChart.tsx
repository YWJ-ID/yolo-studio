import { useEffect, useMemo, useState } from 'react'
import { Empty, Space, Tag, Typography } from 'antd'
import ReactECharts from './echarts'
import type { MetricsSeries } from '../types'

const { Text } = Typography

// 指标展示顺序与配色：检测看 mAP，分类看 top1，损失永远有
const METRIC_ORDER = ['mAP50-95', 'mAP50', 'precision', 'recall', 'f1', 'top1', 'top5', 'train_loss']
const METRIC_COLORS: Record<string, string> = {
  'mAP50-95': '#1677ff',
  'mAP50': '#52c41a',
  precision: '#722ed1',
  recall: '#fa8c16',
  f1: '#13c2c2',
  top1: '#1677ff',
  top5: '#52c41a',
  train_loss: '#cf1322',
}

const DEFAULT_PICK = ['mAP50-95', 'mAP50', 'train_loss']

interface Props {
  series: MetricsSeries | null
  height?: number
}

/** 训练实时曲线。指标来自 results.csv 解析出的 headline，缺失的指标不会出现在图例里。 */
export default function MetricsChart({ series, height = 340 }: Props) {
  const rows = series?.rows ?? []

  const available = useMemo(() => {
    const keys = new Set<string>()
    rows.forEach((r) => Object.keys(r.headline ?? {}).forEach((k) => keys.add(k)))
    const ordered = METRIC_ORDER.filter((k) => keys.has(k))
    keys.forEach((k) => {
      if (!ordered.includes(k)) ordered.push(k)
    })
    return ordered
  }, [rows])

  const [selected, setSelected] = useState<string[]>([])

  // 数据到达后按默认优先级选一次；用户手动改过就不覆盖
  useEffect(() => {
    if (selected.length > 0 || available.length === 0) return
    const pick = DEFAULT_PICK.filter((k) => available.includes(k))
    setSelected(pick.length ? pick : available.slice(0, 3))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [available])

  if (rows.length === 0) {
    return <Empty description="暂无指标（等待 results.csv 写出第一轮）" style={{ padding: 40 }} />
  }

  const xData = rows.map((r) => r.epoch)
  const seriesList = selected
    .filter((k) => available.includes(k))
    .map((key) => ({
      name: key,
      type: 'line' as const,
      smooth: true,
      showSymbol: rows.length <= 40,
      connectNulls: false,
      data: rows.map((r) => (r.headline && key in r.headline ? r.headline[key] : null)),
      lineStyle: { width: 2 },
      itemStyle: { color: METRIC_COLORS[key] ?? '#1677ff' },
    }))

  return (
    <div>
      <Space wrap size={4} style={{ marginBottom: 8 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          显示指标：
        </Text>
        {available.map((k) => {
          const on = selected.includes(k)
          return (
            <Tag
              key={k}
              color={on ? METRIC_COLORS[k] ?? 'blue' : undefined}
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() =>
                setSelected(on ? selected.filter((x) => x !== k) : [...selected, k])
              }
            >
              {k}
            </Tag>
          )
        })}
      </Space>

      <ReactECharts
        style={{ height }}
        option={{
          grid: { left: 8, right: 24, top: 32, bottom: 8, containLabel: true },
          tooltip: { trigger: 'axis' },
          legend: { top: 0, data: seriesList.map((s) => s.name) },
          xAxis: {
            type: 'category',
            name: 'epoch',
            data: xData,
            axisLabel: { fontSize: 11, color: '#595959' },
            axisLine: { lineStyle: { color: '#e8e8e8' } },
          },
          yAxis: {
            type: 'value',
            axisLabel: { fontSize: 11, color: '#595959' },
            splitLine: { lineStyle: { color: '#f5f5f5' } },
          },
          series: seriesList,
        }}
      />
    </div>
  )
}
