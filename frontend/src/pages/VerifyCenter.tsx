import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Input,
  InputNumber,
  Radio,
  Row,
  Select,
  Slider,
  Space,
  Statistic,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  CameraOutlined,
  FileImageOutlined,
  LoadingOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons'
import { api, wsUrl } from '../api/client'
import type { InferFormatInfo, InferFrameResult, InferSessionState } from '../types'

const { Text, Paragraph } = Typography

/** 与 SampleGrid 一致的稳定配色：同一类别在任意画面里颜色相同。 */
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

function categoryColor(name: string, all: string[]): string {
  const idx = all.indexOf(name)
  return PALETTE[(idx < 0 ? 0 : idx) % PALETTE.length]
}

/**
 * 解析类别清单。分隔符支持逗号、分号、竖线、换行（中英文都行）。
 *
 * 只认逗号是不够的：用户很可能顺手打「person;car」——那样整串会被当成**一个**
 * 类名去覆盖某一个下标（真实踩到过），比报错更隐蔽。
 */
export function parseClassList(text: string): string[] {
  return text
    .split(/[,;|、\n\r\t]+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

type Source = 'camera' | 'image' | 'video'

interface FrameStats {
  count: number
  dropped: number
  lastMs: number | null
  /** 最近 N 次的平均毫秒，用于算 FPS */
  window: number[]
}

/** M6 实时验证：摄像头 / 图片 / 视频 → 后端推理 → 画布叠加显示。 */
export default function VerifyCenter() {
  const [formats, setFormats] = useState<InferFormatInfo[]>([])
  const [weights, setWeights] = useState<{ value: string; label: string; source: string; format: string; classes: string[] }[]>([])
  const [session, setSession] = useState<InferSessionState | null>(null)

  const [source, setSource] = useState<Source>('camera')
  const [weightsValue, setWeightsValue] = useState<string>('')
  const [classesText, setClassesText] = useState('')
  const [device, setDevice] = useState('cpu')
  const [imgsz, setImgsz] = useState(640)
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.7)

  const [loadingModel, setLoadingModel] = useState(false)
  const [running, setRunning] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [stats, setStats] = useState<FrameStats>({ count: 0, dropped: 0, lastMs: null, window: [] })
  const [top1Info, setTop1Info] = useState<string>('')

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const offscreenRef = useRef<HTMLCanvasElement | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)
  const inflightRef = useRef(false)
  const frameIdRef = useRef(0)
  const runningRef = useRef(false)
  // drawResult 在闭包里读当前输入源，避免因闭包捕获旧值而画错位置
  const sourceRef = useRef<Source>('camera')
  // 最近一次成功的推理结果：摄像头模式下每帧用它重画叠加层，
  // 这样框能跟着实时画面一起刷新，而不是只在结果到达时才闪一下。
  const lastResultRef = useRef<InferFrameResult | null>(null)
  useEffect(() => {
    sourceRef.current = source
  }, [source])

  const DPR = Math.min(2, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)

  // ---------- 初始加载 ----------

  useEffect(() => {
    api
      .inferFormats()
      .then((res) => {
        setFormats(res.formats)
        setSession(res.active && res.active.loaded ? res.active : null)
      })
      .catch((err) => message.error(err.message ?? '读取推理格式失败'))
    api
      .inferWeights()
      .then((res) => setWeights(res.weights))
      .catch(() => undefined)
    return () => stopAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const formatOf = useCallback(
    (path: string) => weights.find((w) => w.value === path)?.format ?? '',
    [weights],
  )

  // ---------- 会话 ----------

  const loadModel = async () => {
    if (!weightsValue.trim()) {
      message.warning('请先选择或填写权重路径')
      return
    }
    setLoadingModel(true)
    try {
      const res = await api.loadInferModel({
        weights: weightsValue.trim(),
        classes: parseClassList(classesText),
        device,
        imgsz,
        fmt: formatOf(weightsValue.trim()),
      })
      setSession(res.session)
      message.success(`已加载：${res.session.task} / ${res.session.num_classes} 类`)
    } catch (err: any) {
      message.error(err.message ?? '加载权重失败')
    } finally {
      setLoadingModel(false)
    }
  }

  const closeSession = async () => {
    stopAll()
    try {
      await api.closeInferSession()
      setSession(null)
      message.success('已关闭推理会话')
    } catch (err: any) {
      message.error(err.message ?? '关闭失败')
    }
  }

  // ---------- 画面 ----------

  const stopAll = () => {
    runningRef.current = false
    setRunning(false)
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (wsRef.current) {
      try {
        wsRef.current.close()
      } catch {
        /* ignore */
      }
      wsRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
    inflightRef.current = false
  }

  /** 打开摄像头。失败时给出明确原因，不静默。 */
  const openCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      return true
    } catch (err: any) {
      const name = err?.name ?? ''
      const hint =
        name === 'NotAllowedError'
          ? '浏览器拒绝了摄像头权限。请在地址栏左侧的权限设置里允许摄像头后重试。'
          : name === 'NotFoundError'
            ? '没有找到可用的摄像头设备。'
            : name === 'NotReadableError'
              ? '摄像头被其它程序占用（如会议软件），请关闭后重试。'
              : `无法打开摄像头：${err?.message ?? err}`
      setCameraError(hint)
      return false
    }
  }

  /** 把当前视频帧抓成 base64 JPEG（用**离屏** canvas，不碰显示用的 canvas）。 */
  const grabFrame = (): string | null => {
    const video = videoRef.current
    if (!video || video.videoWidth === 0) return null

    // 控制发送尺寸：长边不超过 640，避免带宽与延迟过高
    const maxSide = 640
    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight))
    const w = Math.round(video.videoWidth * scale)
    const h = Math.round(video.videoHeight * scale)

    // 复用同一个离屏 canvas，避免每帧新建
    let off = offscreenRef.current
    if (!off) {
      off = document.createElement('canvas')
      offscreenRef.current = off
    }
    off.width = w
    off.height = h
    const ctx = off.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(video, 0, 0, w, h)
    const dataUrl = off.toDataURL('image/jpeg', 0.7)
    return dataUrl.split(',', 2)[1] ?? null
  }

  /** 把显示用的 canvas 对齐到视频**实际渲染区域**（处理 object-fit: contain 的黑边）。 */
  const syncCanvasToVideo = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.videoWidth === 0) return null

    const rect = video.getBoundingClientRect()
    const vw = video.videoWidth
    const vh = video.videoHeight
    const scale = Math.min(rect.width / vw, rect.height / vh) // contain
    const dispW = Math.max(1, vw * scale)
    const dispH = Math.max(1, vh * scale)

    // 画布像素尺寸（乘 DPR 以在高分屏上清晰）
    const pxW = Math.round(dispW * DPR)
    const pxH = Math.round(dispH * DPR)
    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW
      canvas.height = pxH
    }
    // CSS 尺寸与位置：**必须与 video 里实际显示图片的那块区域重合**，
    // 否则 contain 留下的黑边会让框整体偏移。
    canvas.style.width = `${dispW}px`
    canvas.style.height = `${dispH}px`
    canvas.style.left = `${(rect.width - dispW) / 2}px`
    canvas.style.top = `${(rect.height - dispH) / 2}px`

    return { dispW: pxW, dispH: pxH }
  }, [DPR])

  /** 把一帧结果画到显示画布上（摄像头模式：先画当前帧，再画框）。 */
  const drawResult = useCallback(
    (result: InferFrameResult) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const video = videoRef.current
      const isCamera = sourceRef.current === 'camera'

      // 摄像头模式：画布尺寸对齐视频渲染区，并把当前视频帧画上去作为底图
      let baseW = canvas.width
      let baseH = canvas.height
      if (isCamera) {
        const geo = syncCanvasToVideo()
        if (geo) {
          baseW = geo.dispW
          baseH = geo.dispH
        }
        if (video && video.videoWidth > 0) {
          ctx.setTransform(1, 0, 0, 1, 0, 0)
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        }
      }

      if (result.width && result.height) {
        // 结果坐标基于原图尺寸，换算到画布尺寸
        ctx.setTransform(baseW / result.width, 0, 0, baseH / result.height, 0, 0)
      }

      for (const det of result.detections) {
        const color = categoryColor(det.class_name, result.class_names)
        const [x1, y1, x2, y2] = det.bbox
        ctx.strokeStyle = color
        // 线宽 / 字号按缩放反补，保持视觉一致
        ctx.lineWidth = 2 / (baseW / (result.width || baseW))
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        const label = `${det.class_name} ${det.confidence != null ? det.confidence.toFixed(2) : ''}`.trim()
        const fontPx = 16 / (baseW / (result.width || baseW))
        ctx.font = `${fontPx}px sans-serif`
        const tw = ctx.measureText(label).width
        ctx.fillStyle = color
        ctx.fillRect(x1, Math.max(0, y1 - fontPx * 1.2), tw + 6, fontPx * 1.2)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x1 + 3, Math.max(fontPx, y1 - fontPx * 0.3))
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0)

      if (result.task === 'classify' && result.top1) {
        setTop1Info(`${result.top1.class_name}（${result.top1.confidence ?? '-'}）`)
      }
    },
    [syncCanvasToVideo],
  )

  const recordResult = useCallback((result: InferFrameResult, dropped = 0) => {
    lastResultRef.current = result
    setStats((prev) => {
      const win = [...prev.window, result.duration_ms ?? 0].slice(-10)
      return {
        count: prev.count + 1,
        dropped: prev.dropped + dropped,
        lastMs: result.duration_ms,
        window: win,
      }
    })
  }, [])

  /** 起流：打开 WS，循环抓帧 -> 发送。 */
  const startStream = async () => {
    if (!session) {
      message.warning('请先加载模型')
      return
    }
    if (source === 'camera') {
      const ok = await openCamera()
      if (!ok) return
    }

    const ws = new WebSocket(wsUrl('/api/infer/ws'))
    wsRef.current = ws
    runningRef.current = true
    setRunning(true)
    setStats({ count: 0, dropped: 0, lastMs: null, window: [] })
    setTop1Info('')

    ws.onmessage = (ev) => {
      let msg: any
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      if (msg.type === 'result') {
        inflightRef.current = false
        const result: InferFrameResult = msg.result
        if (result.ok) {
          drawResult(result)
          recordResult(result)
        }
      } else if (msg.type === 'dropped') {
        setStats((prev) => ({ ...prev, dropped: prev.dropped + 1 }))
      } else if (msg.type === 'error') {
        inflightRef.current = false
        message.error(msg.error ?? '推理出错')
      }
    }
    ws.onclose = () => {
      runningRef.current = false
      setRunning(false)
    }
    ws.onerror = () => {
      message.error('WebSocket 连接出错')
    }

    await new Promise<void>((resolve) => {
      if (ws.readyState === WebSocket.OPEN) {
        resolve()
        return
      }
      ws.onopen = () => resolve()
      setTimeout(resolve, 3000)
    })

    const loop = () => {
      if (!runningRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return
      }
      // 摄像头模式：每帧用最近的结果重画叠加层，让框贴着实时画面
      if (sourceRef.current === 'camera' && lastResultRef.current) {
        drawResult(lastResultRef.current)
      }
      // 上一帧还没回来就跳过（丢帧防堆积）
      if (!inflightRef.current) {
        const b64 = grabFrame()
        if (b64) {
          inflightRef.current = true
          frameIdRef.current += 1
          wsRef.current.send(
            JSON.stringify({
              type: 'frame',
              id: frameIdRef.current,
              data: b64,
              options: { conf, iou },
            }),
          )
        }
      }
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
  }

  // ---------- 静态图片 ----------

  const [imagePath, setImagePath] = useState('')
  const [localPreview, setLocalPreview] = useState<string | null>(null)

  const runSingleImage = async () => {
    if (!imagePath.trim()) {
      message.warning('请填写图片路径')
      return
    }
    try {
      const res = await api.inferImage({ path: imagePath.trim(), conf, iou })
      setSession((prev) => prev) // 不变
      // 画到画布
      const img = new Image()
      img.onload = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) return
        ctx.drawImage(img, 0, 0)
        requestAnimationFrame(() => drawResult(res.result))
      }
      img.src = `/api/files/image?path=${encodeURIComponent(imagePath.trim())}`
      recordResult(res.result)
      if (res.result.task === 'classify' && res.result.top1) {
        setTop1Info(`${res.result.top1.class_name}（${res.result.top1.confidence ?? '-'}）`)
      }
    } catch (err: any) {
      message.error(err.message ?? '推理失败')
    }
  }

  /** 本地文件（浏览器侧）直接推理：转 base64 发给后端，无需服务器路径。 */
  const onPickLocalImage = async (file: File) => {
    const reader = new FileReader()
    reader.onload = async () => {
      const dataUrl = String(reader.result)
      setLocalPreview(dataUrl)
      const b64 = dataUrl.split(',', 2)[1]
      try {
        const res = await api.inferImage({ image: b64, conf, iou })
        const img = new Image()
        img.onload = () => {
          const canvas = canvasRef.current
          if (!canvas) return
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          const ctx = canvas.getContext('2d')
          if (!ctx) return
          ctx.drawImage(img, 0, 0)
          requestAnimationFrame(() => drawResult(res.result))
        }
        img.src = dataUrl
        recordResult(res.result)
      } catch (err: any) {
        message.error(err.message ?? '推理失败')
      }
    }
    reader.readAsDataURL(file)
  }

  const fps = useMemo(() => {
    if (stats.window.length < 2) return null
    const avg = stats.window.reduce((a, b) => a + b, 0) / stats.window.length
    return avg > 0 ? 1000 / avg : null
  }, [stats.window])

  const availableFormats = formats.filter((f) => f.available)
  const selectedFormatInfo = formats.find((f) => f.name === formatOf(weightsValue))

  return (
    <div className="page">
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="实时验证：用模型看它在新画面上认出了什么"
        description="这里只做「显示」，不产出数据集、也不计算指标。评估指标（mAP 等）请用模型库里的评估功能。摄像头画面只在你的浏览器与后端之间流转，不会被保存。"
      />

      <Row gutter={[16, 16]}>
        {/* 左：配置 */}
        <Col xs={24} lg={9}>
          <Card title="模型与参数" size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">权重来源</Text>
                <Select
                  style={{ width: '100%', marginTop: 4 }}
                  placeholder="从模型库 / 导出产物选择"
                  value={weightsValue || undefined}
                  onChange={(v) => setWeightsValue(v)}
                  options={weights.map((w) => ({ value: w.value, label: w.label }))}
                  notFoundContent={<Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无可选项" />}
                />
                <Input
                  style={{ marginTop: 8 }}
                  placeholder="或直接填写任意权重路径（.pt / .onnx / .torchscript）"
                  value={weightsValue}
                  onChange={(e) => setWeightsValue(e.target.value)}
                />
              </div>

              {selectedFormatInfo && (
                <Alert
                  type={selectedFormatInfo.available ? 'success' : 'warning'}
                  showIcon
                  message={`格式：${selectedFormatInfo.label}`}
                  description={selectedFormatInfo.available ? selectedFormatInfo.task_hint : selectedFormatInfo.reason}
                />
              )}

              {formatOf(weightsValue) === 'onnx' && (
                <Alert
                  type="info"
                  showIcon
                  message="ONNX 的类名不一定有"
                  description="ultralytics 导出的 ONNX 通常会把类别名写进文件（可自动读出）；若没有，就必须在下面填写，否则类名会显示成 class_0 / class_1。加载后如果类别显示异常，回来补填再加载一次即可。"
                />
              )}

              <div>
                <Text type="secondary">类别清单（逗号分隔；覆盖模型自带的类名）</Text>
                <Input
                  style={{ marginTop: 4 }}
                  placeholder="如：person,car,dog"
                  value={classesText}
                  onChange={(e) => setClassesText(e.target.value)}
                />
              </div>

              <Row gutter={8}>
                <Col span={12}>
                  <Text type="secondary">设备</Text>
                  <Select
                    style={{ width: '100%', marginTop: 4 }}
                    value={device}
                    onChange={setDevice}
                    options={[
                      { value: 'cpu', label: 'CPU' },
                      { value: '0', label: 'GPU 0' },
                    ]}
                  />
                </Col>
                <Col span={12}>
                  <Text type="secondary">推理尺寸</Text>
                  <InputNumber
                    style={{ width: '100%', marginTop: 4 }}
                    min={32}
                    max={1920}
                    step={32}
                    value={imgsz}
                    onChange={(v) => setImgsz(Number(v) || 640)}
                  />
                </Col>
              </Row>

              <div>
                <Text type="secondary">置信度阈值 conf：{conf.toFixed(2)}</Text>
                <Slider min={0.01} max={1} step={0.01} value={conf} onChange={setConf} />
              </div>
              <div>
                <Text type="secondary">NMS IoU：{iou.toFixed(2)}</Text>
                <Slider min={0.05} max={0.95} step={0.05} value={iou} onChange={setIou} />
              </div>

              <Space>
                <Button type="primary" loading={loadingModel} onClick={loadModel}>
                  加载模型
                </Button>
                <Button onClick={closeSession} disabled={!session}>
                  关闭会话
                </Button>
              </Space>

              {session && (
                <Descriptions size="small" column={1} bordered>
                  <Descriptions.Item label="任务">{session.task}</Descriptions.Item>
                  <Descriptions.Item label="类别数">{session.num_classes}</Descriptions.Item>
                  <Descriptions.Item label="类别">
                    {session.classes.length ? (
                      session.classes.map((c) => (
                        <Tag key={c} color={categoryColor(c, session.classes)}>
                          {c}
                        </Tag>
                      ))
                    ) : (
                      <Text type="secondary">模型没带类名，且未填写——框上会显示 class_0 / class_1</Text>
                    )}
                  </Descriptions.Item>
                  {session.classes.length > 0 && (
                    <Descriptions.Item label="类名来源">
                      {session.class_source === 'request' ? (
                        <Text>你在下方填写的</Text>
                      ) : (
                        <Text>模型文件自带</Text>
                      )}
                    </Descriptions.Item>
                  )}
                </Descriptions>
              )}

              <div>
                <Text type="secondary">可推理格式与本机可用性</Text>
                <div style={{ marginTop: 6 }}>
                  <Space wrap size={[6, 6]}>
                    {formats.map((f) => (
                      <Tag key={f.name} color={f.available ? 'green' : 'default'}>
                        {f.label}
                        {!f.available && ` — ${f.reason}`}
                      </Tag>
                    ))}
                  </Space>
                </div>
                <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 6, marginBottom: 0 }}>
                  本机可用 {availableFormats.length} / {formats.length} 种。
                </Paragraph>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 右：画面 */}
        <Col xs={24} lg={15}>
          <Card
            title="验证画面"
            size="small"
            extra={
              <Space>
                <Radio.Group value={source} onChange={(e) => setSource(e.target.value)} size="small">
                  <Radio.Button value="camera">
                    <CameraOutlined /> 摄像头
                  </Radio.Button>
                  <Radio.Button value="image">
                    <FileImageOutlined /> 图片
                  </Radio.Button>
                  <Radio.Button value="video">
                    <VideoCameraOutlined /> 视频
                  </Radio.Button>
                </Radio.Group>
              </Space>
            }
          >
            {source === 'camera' && (
              <Alert
                type="warning"
                showIcon
                style={{ marginBottom: 12 }}
                message="画面不会被保存"
                description="摄像头画面只在本机浏览器与后端之间流转，不落盘、不外传。"
              />
            )}

            {cameraError && (
              <Alert type="error" showIcon style={{ marginBottom: 12 }} message="无法打开摄像头" description={cameraError} />
            )}

            {source === 'camera' && (
              <Space style={{ marginBottom: 12 }}>
                <Button
                  type="primary"
                  icon={running ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                  onClick={() => (running ? stopAll() : startStream())}
                  disabled={!session}
                >
                  {running ? '停止' : '开始'}
                </Button>
                {!session && <Text type="secondary">请先加载模型</Text>}
              </Space>
            )}

            {source === 'image' && (
              <Space direction="vertical" style={{ width: '100%', marginBottom: 12 }}>
                <Space.Compact style={{ width: '100%' }}>
                  <Input
                    placeholder="服务器上的图片路径"
                    value={imagePath}
                    onChange={(e) => setImagePath(e.target.value)}
                  />
                  <Button type="primary" onClick={runSingleImage} disabled={!session}>
                    推理
                  </Button>
                </Space.Compact>
                <Space>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (f) onPickLocalImage(f)
                    }}
                  />
                  <Text type="secondary">或选择本地图片（直接上传，不需要服务器路径）</Text>
                </Space>
              </Space>
            )}

            {source === 'video' && (
              <Alert
                type="info"
                showIcon
                style={{ marginBottom: 12 }}
                message="视频验证暂未单独实现"
                description="当前版本可先用摄像头做实时验证，或对逐帧导出的图片用「图片」模式批量看。视频流的逐帧送检会放在后续任务里。"
              />
            )}

            {/*
              摄像头模式：视频铺底、canvas 绝对定位叠在**同一位置**透明显示（只画框）。
              图片模式：只用 canvas（图片直接画上去）。
              关键：抓帧与画框用的是同一个 canvas，摄像头模式下它必须可见，
              否则框会被画进一个 display:none 的元素里，画面上什么都看不到。
            */}
            <div
              style={{
                position: 'relative',
                background: '#000',
                borderRadius: 6,
                minHeight: 360,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <video
                ref={videoRef}
                style={{
                  display: source === 'camera' ? 'block' : 'none',
                  width: '100%',
                  maxHeight: 480,
                  objectFit: 'contain',
                }}
                playsInline
                muted
              />
              <canvas
                ref={canvasRef}
                style={
                  source === 'camera'
                    ? {
                        // 叠在视频之上；尺寸与 video 的显示尺寸一致，指针事件穿透
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                      }
                    : { maxWidth: '100%', maxHeight: 480 }
                }
              />
            </div>

            {/* 实时统计 */}
            <Row gutter={16} style={{ marginTop: 12 }}>
              <Col span={6}>
                <Statistic title="已推理帧" value={stats.count} valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col span={6}>
                <Statistic
                  title="单帧耗时"
                  value={stats.lastMs != null ? stats.lastMs : '-'}
                  suffix="ms"
                  valueStyle={{ fontSize: 16 }}
                />
              </Col>
              <Col span={6}>
                <Statistic title="约 FPS" value={fps != null ? fps.toFixed(1) : '-'} valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col span={6}>
                <Statistic title="丢弃帧" value={stats.dropped} valueStyle={{ fontSize: 16 }} />
              </Col>
            </Row>

            {top1Info && (
              <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
                <Tag color="blue">分类结果</Tag>
                {top1Info}
              </Paragraph>
            )}

            <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
              CPU 上单帧约 30~150 ms，因此帧率通常只有几 FPS 到十几 FPS——够「肉眼验证识别情况」，
              但不足以当流畅视频看。上一帧没回来时会跳过新帧（淘汰旧帧防延迟堆积），
              「丢弃帧」统计的就是这种情况。
            </Paragraph>
          </Card>
        </Col>
      </Row>
    </div>
  )
}
