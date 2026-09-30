import { Alert, Card, Col, Descriptions, Empty, Row, Space, Statistic, Table, Tag, Typography } from 'antd'
import ReactECharts from './echarts'
import type { EvalResult } from '../types'
import { formatMetric } from '../utils'

const { Text } = Typography

const OVERALL_ORDER = ['mAP50-95', 'mAP50', 'precision', 'recall', 'f1', 'top1', 'top5', 'fitness']
const OVERALL_LABELS: Record<string, string> = {
  'mAP50-95': 'mAP50-95',
  'mAP50': 'mAP50',
  precision: 'precision',
  recall: 'recall',
  f1: 'F1',
  top1: 'top1',
  top5: 'top5',
  fitness: 'fitness',
}

/** 单次评估结果的完整视图：总体指标 + 逐类指标 + 混淆矩阵 + 速度。 */
export default function EvalResultView({ result }: { result: EvalResult }) {
  const overallKeys = OVERALL_ORDER.filter((k) => result.overall[k] != null)

  return (
    <div>
      {!result.ok && (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 12 }}
          message="本次评估未成功完成，指标可能缺失"
          description={result.error || undefined}
        />
      )}

      <Space wrap size={16}>
        {overallKeys.length ? (
          overallKeys.map((k) => (
            <Statistic
              key={k}
              title={OVERALL_LABELS[k] ?? k}
              value={result.overall[k]}
              precision={4}
              valueStyle={{ fontSize: 18 }}
            />
          ))
        ) : (
          <Text type="secondary">没有可解析的总体指标</Text>
        )}
      </Space>

      <Descriptions column={{ xs: 1, md: 2 }} size="small" style={{ marginTop: 12 }}>
        <Descriptions.Item label="划分">
          <Tag color="geekblue">{result.split || '-'}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="任务">{result.task}</Descriptions.Item>
        <Descriptions.Item label="权重">
          <Text className="mono" style={{ fontSize: 11 }}>
            {result.weights || '-'}
          </Text>
        </Descriptions.Item>
        <Descriptions.Item label="数据">
          <Text className="mono" style={{ fontSize: 11 }}>
            {result.data_yaml || '-'}
          </Text>
        </Descriptions.Item>
        <Descriptions.Item label="耗时">{result.duration_sec} s</Descriptions.Item>
        <Descriptions.Item label="评估时间">{result.created_at}</Descriptions.Item>
      </Descriptions>

      {Object.keys(result.speed).length > 0 && (
        <Space wrap style={{ marginTop: 8 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            速度（每图毫秒）：
          </Text>
          {Object.entries(result.speed).map(([k, v]) => (
            <Tag key={k}>
              {k} {formatMetric(v, 2)}
            </Tag>
          ))}
        </Space>
      )}

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={result.confusion_matrix ? 14 : 24}>
          <Card title={`逐类指标（${result.per_class.length} 类）`} size="small">
            {result.per_class.length ? (
              <Table
                size="small"
                rowKey="name"
                pagination={false}
                scroll={{ x: 'max-content', y: 320 }}
                dataSource={result.per_class}
                columns={[
                  { title: '#', dataIndex: 'index', width: 50 },
                  { title: '类别', dataIndex: 'name', fixed: 'left' as const },
                  { title: '实例数', dataIndex: 'instances', width: 80, sorter: (a, b) => a.instances - b.instances },
                  { title: 'P', dataIndex: 'precision', width: 90, render: (v: number) => formatMetric(v) },
                  { title: 'R', dataIndex: 'recall', width: 90, render: (v: number) => formatMetric(v) },
                  { title: 'F1', dataIndex: 'f1', width: 90, render: (v: number) => formatMetric(v) },
                  { title: 'AP50', dataIndex: 'ap50', width: 90, render: (v: number) => formatMetric(v) },
                  { title: 'AP50-95', dataIndex: 'ap50_95', width: 100, render: (v: number) => formatMetric(v) },
                ]}
              />
            ) : (
              <Empty description="没有逐类指标" />
            )}
          </Card>
        </Col>

        {result.confusion_matrix && (
          <Col xs={24} lg={10}>
            <Card
              title="混淆矩阵"
              size="small"
              extra={<Text type="secondary" style={{ fontSize: 11 }}>{result.confusion_matrix.axis}</Text>}
            >
              <ConfusionMatrixChart
                labels={result.confusion_matrix.labels}
                matrix={result.confusion_matrix.matrix}
              />
            </Card>
          </Col>
        )}
      </Row>
    </div>
  )
}

function ConfusionMatrixChart({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const data: [number, number, number][] = []
  let max = 0
  matrix.forEach((row, i) =>
    row.forEach((v, j) => {
      data.push([j, i, v])
      if (v > max) max = v
    }),
  )

  return (
    <ReactECharts
      style={{ height: 340 }}
      option={{
        grid: { left: 8, right: 24, top: 16, bottom: 40, containLabel: true },
        tooltip: {
          position: 'top',
          formatter: (p: any) =>
            `真实 ${labels[p.value[1]] ?? '?'} → 预测 ${labels[p.value[0]] ?? '?'}: ${p.value[2]}`,
        },
        xAxis: {
          type: 'category',
          data: labels,
          name: '预测',
          axisLabel: { fontSize: 10, rotate: 30, color: '#595959' },
        },
        yAxis: {
          type: 'category',
          data: labels,
          name: '真实',
          axisLabel: { fontSize: 10, color: '#595959' },
        },
        visualMap: {
          min: 0,
          max: Math.max(1, max),
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: 0,
          inRange: { color: ['#f5f5f5', '#1677ff', '#003a8c'] },
          textStyle: { fontSize: 10 },
        },
        series: [
          {
            type: 'heatmap',
            data,
            label: { show: max > 0 && labels.length <= 12, fontSize: 10 },
            emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.3)' } },
          },
        ],
      }}
    />
  )
}
