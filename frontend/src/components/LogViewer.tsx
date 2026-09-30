import { useEffect, useRef, useState } from 'react'
import { Button, Space, Switch, Typography } from 'antd'
import { ClearOutlined, VerticalAlignBottomOutlined } from '@ant-design/icons'
import type { LogLine } from '../types'

const { Text } = Typography

interface Props {
  lines: LogLine[]
  height?: number
  connected?: boolean
  onClear?: () => void
  emptyText?: string
}

/** 网页终端：训练 / 评估 / 导出的日志都是同一结构，直接复用。 */
export default function LogViewer({ lines, height = 320, connected, onClear, emptyText }: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (!autoScroll) return
    const el = boxRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines, autoScroll])

  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <Space size={8}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {lines.length} 行
          </Text>
          {connected != null && (
            <Text type={connected ? 'success' : 'secondary'} style={{ fontSize: 12 }}>
              {connected ? '● 已连接' : '○ 未连接'}
            </Text>
          )}
        </Space>
        <Space size={8}>
          <Space size={4}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              自动滚动
            </Text>
            <Switch size="small" checked={autoScroll} onChange={setAutoScroll} />
          </Space>
          <Button
            size="small"
            icon={<VerticalAlignBottomOutlined />}
            onClick={() => {
              const el = boxRef.current
              if (el) el.scrollTop = el.scrollHeight
            }}
          >
            到底部
          </Button>
          {onClear && (
            <Button size="small" icon={<ClearOutlined />} onClick={onClear}>
              清空
            </Button>
          )}
        </Space>
      </div>
      <div ref={boxRef} className="log-body" style={{ height }}>
        {lines.length === 0 ? (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {emptyText ?? '暂无日志'}
          </Text>
        ) : (
          lines.map((l) => (
            <div key={l.seq} className="log-line">
              <span className="log-seq">{l.seq}</span>
              <span className="log-text">{l.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
