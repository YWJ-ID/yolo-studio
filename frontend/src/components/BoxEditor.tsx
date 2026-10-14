import { useCallback, useEffect, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { Typography } from 'antd'
import { imageUrl } from '../api/client'
import type { SampleImage } from '../types'
import { categoryColor } from './SampleGrid'

const { Text } = Typography

type Box = [number, number, number, number]
type Corner = 'nw' | 'ne' | 'sw' | 'se'

interface Gesture {
  kind: 'move' | 'resize' | 'draw'
  /** move / resize 作用于这个标注 */
  id?: string
  corner?: Corner
  /** 起始框（move / resize） */
  orig?: Box
  startX: number
  startY: number
  curX: number
  curY: number
}

interface Props {
  sample: SampleImage
  categories: string[]
  /** 新增框使用的类别 */
  activeCategory: string
  /** select = 选中/移动/缩放；draw = 拖拽画新框 */
  tool: 'select' | 'draw'
  selectedId: string | null
  busy?: boolean
  onSelect: (id: string | null) => void
  onAdd: (bbox: Box) => void
  onUpdate: (id: string, patch: { bbox?: Box }) => void
}

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

function normalize(a: number, b: number): [number, number] {
  return a <= b ? [a, b] : [b, a]
}

/** 在一张图上叠加可交互的标注框：选中、拖动、四角缩放、拖拽画新框。

    坐标换算：容器按图像宽高比布局、SVG 用 preserveAspectRatio="none"，
    因此鼠标位置到图像像素是线性映射（避免 contain 黑边造成的偏移）。
*/
export default function BoxEditor({
  sample,
  categories,
  activeCategory,
  tool,
  selectedId,
  busy,
  onSelect,
  onAdd,
  onUpdate,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [gesture, setGesture] = useState<Gesture | null>(null)

  const W = sample.width || 1
  const H = sample.height || 1
  const handleSize = Math.max(6, Math.min(W, H) * 0.02)
  const strokeW = 2
  const fontSize = Math.max(10, W * 0.018)

  // 图像或工具变化时丢弃进行中的手势
  useEffect(() => {
    setGesture(null)
  }, [sample.uid, tool])

  const toImage = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect || rect.width === 0 || rect.height === 0) return [0, 0]
      const x = ((clientX - rect.left) / rect.width) * W
      const y = ((clientY - rect.top) / rect.height) * H
      return [clamp(x, 0, W), clamp(y, 0, H)]
    },
    [W, H],
  )

  /** 由手势算出当前应显示的框。 */
  const computeBox = (g: Gesture): Box | null => {
    if (g.kind === 'draw') {
      const [x1, x2] = normalize(g.startX, g.curX)
      const [y1, y2] = normalize(g.startY, g.curY)
      return [x1, y1, x2, y2]
    }
    const orig = g.orig
    if (!orig) return null
    if (g.kind === 'move') {
      const dx = g.curX - g.startX
      const dy = g.curY - g.startY
      const w = orig[2] - orig[0]
      const h = orig[3] - orig[1]
      const nx1 = clamp(orig[0] + dx, 0, W - w)
      const ny1 = clamp(orig[1] + dy, 0, H - h)
      return [nx1, ny1, nx1 + w, ny1 + h]
    }
    let [x1, y1, x2, y2] = orig
    if (g.corner === 'nw') {
      x1 = g.curX
      y1 = g.curY
    } else if (g.corner === 'ne') {
      x2 = g.curX
      y1 = g.curY
    } else if (g.corner === 'sw') {
      x1 = g.curX
      y2 = g.curY
    } else {
      x2 = g.curX
      y2 = g.curY
    }
    const [nx1, nx2] = normalize(x1, x2)
    const [ny1, ny2] = normalize(y1, y2)
    return [nx1, ny1, nx2, ny2]
  }

  const onPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (busy) return
    const target = e.target as SVGElement
    const boxId = target.dataset?.boxId
    const handle = target.dataset?.handle as Corner | undefined
    const [x, y] = toImage(e.clientX, e.clientY)
    e.preventDefault()

    if (tool === 'draw') {
      setGesture({ kind: 'draw', startX: x, startY: y, curX: x, curY: y })
      svgRef.current?.setPointerCapture(e.pointerId)
      return
    }
    if (boxId) {
      const obj = sample.objects.find((o) => o.box_id === boxId)
      if (obj?.bbox) {
        onSelect(boxId)
        setGesture(
          handle
            ? {
                kind: 'resize',
                id: boxId,
                corner: handle,
                orig: obj.bbox as Box,
                startX: x,
                startY: y,
                curX: x,
                curY: y,
              }
            : { kind: 'move', id: boxId, orig: obj.bbox as Box, startX: x, startY: y, curX: x, curY: y },
        )
        svgRef.current?.setPointerCapture(e.pointerId)
      }
      return
    }
    onSelect(null)
  }

  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!gesture) return
    const [x, y] = toImage(e.clientX, e.clientY)
    setGesture((g) => (g ? { ...g, curX: x, curY: y } : g))
  }

  const onPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!gesture) return
    const [x, y] = toImage(e.clientX, e.clientY)
    const g = { ...gesture, curX: x, curY: y }
    setGesture(null)
    const box = computeBox(g)
    if (!box) return
    if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return
    if (g.kind === 'draw') {
      onAdd(box)
    } else if (g.id) {
      onUpdate(g.id, { bbox: box })
    }
  }

  const draftBox = gesture?.kind === 'draw' ? computeBox(gesture) : null

  return (
    <div
      className="box-editor"
      style={{ position: 'relative', width: '100%', aspectRatio: `${W} / ${H}`, background: '#1f1f1f' }}
    >
      <img
        src={imageUrl(sample.path)}
        alt={sample.rel_path}
        draggable={false}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          display: 'block',
          userSelect: 'none',
        }}
      />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          cursor: busy ? 'wait' : tool === 'draw' ? 'crosshair' : 'default',
          touchAction: 'none',
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => setGesture(null)}
      >
        {sample.objects.map((obj) => {
          if (!obj.bbox) return null
          const id = obj.box_id || `${obj.category}-${obj.bbox.join(',')}`
          const live =
            gesture && gesture.kind !== 'draw' && gesture.id === obj.box_id ? computeBox(gesture) : null
          const b = live ?? (obj.bbox as Box)
          const selected = selectedId === id
          const color = categoryColor(obj.category, categories)
          const human = obj.source === 'human_review' || obj.review === 'added' || obj.review === 'edited'
          const label = human ? `${obj.category} *` : obj.category
          const w = Math.max(label.length * fontSize * 0.6, fontSize * 2)
          return (
            <g key={id}>
              <rect
                data-box-id={obj.box_id}
                x={b[0]}
                y={b[1]}
                width={Math.max(0, b[2] - b[0])}
                height={Math.max(0, b[3] - b[1])}
                fill={selected ? `${color}22` : 'transparent'}
                stroke={color}
                strokeWidth={selected ? strokeW + 1 : strokeW}
                strokeDasharray={obj.source === 'model_prediction' && !human ? undefined : '6 4'}
                vectorEffect="non-scaling-stroke"
                style={{ cursor: tool === 'draw' ? 'crosshair' : 'move' }}
              />
              <rect
                x={b[0]}
                y={Math.max(0, b[1] - fontSize * 1.35)}
                width={w}
                height={fontSize * 1.35}
                fill={color}
                opacity={0.85}
                pointerEvents="none"
              />
              <text
                x={b[0] + fontSize * 0.25}
                y={Math.max(fontSize, b[1] - fontSize * 0.35)}
                fontSize={fontSize}
                fill="#fff"
                pointerEvents="none"
              >
                {label}
              </text>
              {selected &&
                !busy &&
                (['nw', 'ne', 'sw', 'se'] as Corner[]).map((c) => {
                  const hx = c === 'nw' || c === 'sw' ? b[0] : b[2]
                  const hy = c === 'nw' || c === 'ne' ? b[1] : b[3]
                  return (
                    <rect
                      key={c}
                      data-box-id={obj.box_id}
                      data-handle={c}
                      x={hx - handleSize / 2}
                      y={hy - handleSize / 2}
                      width={handleSize}
                      height={handleSize}
                      fill="#fff"
                      stroke={color}
                      strokeWidth={1.5}
                      vectorEffect="non-scaling-stroke"
                      style={{ cursor: `${c}-resize` }}
                    />
                  )
                })}
            </g>
          )
        })}
        {draftBox && (
          <rect
            x={draftBox[0]}
            y={draftBox[1]}
            width={Math.max(0, draftBox[2] - draftBox[0])}
            height={Math.max(0, draftBox[3] - draftBox[1])}
            fill={`${categoryColor(activeCategory, categories)}22`}
            stroke={categoryColor(activeCategory, categories)}
            strokeWidth={strokeW}
            strokeDasharray="6 4"
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
        )}
      </svg>
      {busy && (
        <div className="box-editor-busy">
          <Text style={{ color: '#fff' }}>保存中…</Text>
        </div>
      )}
    </div>
  )
}
