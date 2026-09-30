import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '../api/client'

/**
 * 订阅一个任务（训练 / 评估 / 导出）的 WebSocket 增量事件。
 *
 * path 传后端 WS 路径（如 /api/train/jobs/xxx/ws），为 null 时不连接。
 * 事件回调放进 ref，避免因调用方每次渲染新建函数而反复重连。
 * 服务端关闭码 4404 表示任务不存在，此时不再重连。
 */
export function useJobSocket<T extends { type: string }>(
  path: string | null,
  onEvent: (event: T) => void,
): { connected: boolean; closed: boolean } {
  const [connected, setConnected] = useState(false)
  const [closed, setClosed] = useState(false)
  const handlerRef = useRef(onEvent)
  handlerRef.current = onEvent

  useEffect(() => {
    setClosed(false)
    if (!path) return
    let ws: WebSocket | null = null
    let retry: number | undefined

    const connect = () => {
      ws = new WebSocket(wsUrl(path))
      ws.onopen = () => setConnected(true)
      ws.onerror = () => setConnected(false)
      ws.onclose = (e) => {
        setConnected(false)
        if (e.code === 4404) {
          setClosed(true)
          return
        }
        // 网络抖动或后端重启：稍后重连，直到组件卸载
        retry = window.setTimeout(connect, 2000)
      }
      ws.onmessage = (e) => {
        try {
          handlerRef.current(JSON.parse(e.data) as T)
        } catch {
          // 忽略无法解析的帧
        }
      }
    }

    connect()
    return () => {
      if (retry) window.clearTimeout(retry)
      ws?.close()
    }
  }, [path])

  return { connected, closed }
}
