import { Card, Col, Image, Row, Tag, Typography } from 'antd'
import { imageUrl } from '../api/client'
import type { SampleImage } from '../types'

const { Text } = Typography

/** 稳定配色：同一类别在任意图上颜色一致。 */
const PALETTE = [
  '#f5222d',
  '#fa8c16',
  '#faad14',
  '#52c41a',
  '#13c2c2',
  '#1677ff',
  '#722ed1',
  '#eb2f96',
]

export function categoryColor(category: string, all: string[]): string {
  const idx = all.indexOf(category)
  return PALETTE[(idx < 0 ? 0 : idx) % PALETTE.length]
}

interface Props {
  sample: SampleImage
  categories: string[]
}

/** 单张样本图 + 标注叠加预览。

    检测数据（kind=bbox）画框；分类数据（kind=image）无框，只显示类别标签。
*/
export default function SampleCard({ sample, categories }: Props) {
  const scale = 220 / (sample.width || 1)
  const boxed = sample.objects.filter((o) => o.bbox)

  return (
    <Card
      size="small"
      cover={
        <div className="sample-canvas" style={{ height: sample.height * scale }}>
          <Image
            src={imageUrl(sample.path)}
            alt={sample.rel_path}
            preview={false}
            width="100%"
            height="100%"
            style={{ objectFit: 'contain' }}
          />
          <svg
            className="sample-overlay"
            viewBox={`0 0 ${sample.width} ${sample.height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {boxed.map((obj, i) => {
              const b = obj.bbox as [number, number, number, number]
              return (
                <rect
                  key={i}
                  x={b[0]}
                  y={b[1]}
                  width={Math.max(0, b[2] - b[0])}
                  height={Math.max(0, b[3] - b[1])}
                  fill="none"
                  stroke={categoryColor(obj.category, categories)}
                  strokeWidth={Math.max(1, sample.width / 300)}
                />
              )
            })}
          </svg>
        </div>
      }
    >
      <Text className="sample-path" ellipsis={{ tooltip: sample.rel_path }}>
        {sample.rel_path}
      </Text>
      <div className="sample-meta">
        <Text type="secondary">
          {sample.width}×{sample.height}
        </Text>
        {sample.split && <Tag color="geekblue">{sample.split}</Tag>}
        {sample.group && (
          <Tag color="purple" title={sample.group}>
            组: {sample.group.length > 14 ? `${sample.group.slice(0, 14)}…` : sample.group}
          </Tag>
        )}
        <Tag>{sample.num_objects} 个目标</Tag>
      </div>
      <div className="sample-tags">
        {[...new Set(sample.objects.map((o) => o.category))].map((c) => (
          <Tag key={c} color={categoryColor(c, categories)}>
            {c}
          </Tag>
        ))}
      </div>
    </Card>
  )
}

export function SampleGrid({ samples, categories }: { samples: SampleImage[]; categories: string[] }) {
  return (
    <Row gutter={[12, 12]}>
      {samples.map((s) => (
        <Col key={s.uid} xs={24} sm={12} md={8} xl={6}>
          <SampleCard sample={s} categories={categories} />
        </Col>
      ))}
    </Row>
  )
}
