import { Empty, Image, Tabs, Tag, Typography } from 'antd'
import type { ArtifactsResponse } from '../types'
import { formatBytes } from '../utils'

const { Text } = Typography

/** 训练 / 评估过程图像分组浏览。后端已在每张图上拼好受限访问的 url。 */
export default function ArtifactGallery({ data, height = 150 }: { data: ArtifactsResponse; height?: number }) {
  if (!data.images?.length) {
    return <Empty description="暂无过程图像（训练开始后会逐步生成）" />
  }

  const groups = data.groups?.length
    ? data.groups
    : [{ id: 'all', label: '全部', count: data.images.length }]

  return (
    <Tabs
      size="small"
      items={groups.map((g) => {
        const items = data.images.filter((i) => i.group === g.id)
        return {
          key: g.id,
          label: (
            <span>
              {g.label} <Tag style={{ marginInlineStart: 4 }}>{g.count}</Tag>
            </span>
          ),
          children: items.length ? (
            <Image.PreviewGroup>
              <div className="artifact-grid">
                {items.map((img) => (
                  <div key={img.name} className="artifact-item">
                    <Image
                      src={img.url}
                      alt={img.name}
                      height={height}
                      style={{ objectFit: 'cover', width: '100%' }}
                    />
                    <Text className="sample-path" ellipsis={{ tooltip: img.name }}>
                      {img.name}
                    </Text>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {formatBytes(img.size)}
                    </Text>
                  </div>
                ))}
              </div>
            </Image.PreviewGroup>
          ) : (
            <Empty description="该分组暂无图像" />
          ),
        }
      })}
    />
  )
}
