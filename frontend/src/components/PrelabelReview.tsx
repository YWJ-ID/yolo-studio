import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Col,
  Empty,
  Progress,
  Row,
  Segmented,
  Select,
  Space,
  Statistic,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  CheckCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  LeftOutlined,
  ReloadOutlined,
  RightOutlined,
  UndoOutlined,
} from '@ant-design/icons'
import { api, imageUrl } from '../api/client'
import type { PrelabelJob, PrelabelReviewSummary, SampleImage } from '../types'
import BoxEditor from './BoxEditor'
import { categoryColor } from './SampleGrid'

const { Text } = Typography

const PAGE_SIZE = 24

const FILTERS = [
  { value: 'pending', label: '待复核' },
  { value: 'edited', label: '我改过的' },
  { value: 'empty', label: '无任何框' },
  { value: 'with_boxes', label: '有框' },
  { value: 'confirmed', label: '已确认' },
  { value: 'all', label: '全部' },
]

interface Props {
  job: PrelabelJob
  categories: string[]
  /** 复核后通知父级刷新（如更新复核进度） */
  onChanged?: () => void
}

/** M7-04 复核 / 编辑：逐张修正伪标签。 */
export default function PrelabelReview({ job, categories, onChanged }: Props) {
  const [filter, setFilter] = useState('pending')
  const [page, setPage] = useState(0)
  const [samples, setSamples] = useState<SampleImage[]>([])
  const [total, setTotal] = useState(0)
  const [summary, setSummary] = useState<PrelabelReviewSummary | null>(job.review)
  const [index, setIndex] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [tool, setTool] = useState<'select' | 'draw'>('select')
  const [activeCategory, setActiveCategory] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)

  const declaredCategories = categories.length ? categories : job.classes
  const current = samples[index] as SampleImage | undefined
  const categoryOptions = useMemo(
    () => (declaredCategories.length ? declaredCategories : []).map((c) => ({ value: c })),
    [declaredCategories],
  )

  useEffect(() => {
    if (!activeCategory && declaredCategories.length) setActiveCategory(declaredCategories[0])
  }, [declaredCategories, activeCategory])

  const load = useCallback(
    async (targetFilter = filter, targetPage = page) => {
      setLoading(true)
      try {
        const res = await api.prelabelReviewSamples(job.id, targetFilter, targetPage * PAGE_SIZE, PAGE_SIZE)
        setSamples(res.images)
        setTotal(res.total)
        setSummary(res.summary)
        setIndex((i) => Math.min(i, Math.max(0, res.images.length - 1)))
        setSelectedId(null)
      } catch (err: any) {
        message.error(err.message ?? '读取复核样本失败')
      } finally {
        setLoading(false)
      }
    },
    [job.id, filter, page],
  )

  useEffect(() => {
    void load(filter, page)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter, page, job.id])

  const applyResult = useCallback(
    (image: SampleImage, nextSummary: PrelabelReviewSummary) => {
      setSummary(nextSummary)
      setSamples((list) => list.map((s) => (s.uid === image.uid ? image : s)))
      onChanged?.()
    },
    [onChanged],
  )

  const run = useCallback(
    async (fn: () => Promise<{ image: SampleImage; summary: PrelabelReviewSummary }>) => {
      setBusy(true)
      try {
        const res = await fn()
        applyResult(res.image, res.summary)
      } catch (err: any) {
        message.error(err.message ?? '操作失败')
      } finally {
        setBusy(false)
      }
    },
    [applyResult],
  )

  const goNext = useCallback(() => {
    setSelectedId(null)
    setIndex((i) => {
      if (i + 1 < samples.length) return i + 1
      if ((page + 1) * PAGE_SIZE < total) setPage((p) => p + 1)
      return i
    })
  }, [samples.length, page, total])

  const goPrev = useCallback(() => {
    setSelectedId(null)
    setIndex((i) => {
      if (i - 1 >= 0) return i - 1
      if (page > 0) setPage((p) => p - 1)
      return i
    })
  }, [page])

  const confirmCurrent = useCallback(
    async (status: 'confirmed' | 'pending') => {
      if (!current) return
      await run(() => api.setPrelabelImageStatus(job.id, current.uid, status))
      if (status === 'confirmed') goNext()
    },
    [current, job.id, run, goNext],
  )

  const deleteSelected = useCallback(async () => {
    if (!current || !selectedId) return
    await run(() => api.deletePrelabelAnnotation(job.id, selectedId))
    setSelectedId(null)
  }, [current, job.id, selectedId, run])

  // 键盘快捷键（焦点在输入控件上时让位给控件）
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return
      if (busy || !current) return
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        goNext()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goPrev()
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedId) {
          e.preventDefault()
          void deleteSelected()
        }
      } else if (e.key === 'Enter') {
        e.preventDefault()
        void confirmCurrent('confirmed')
      } else if (e.key === 'Escape') {
        setSelectedId(null)
      } else if (e.key === 'd') {
        setTool('draw')
      } else if (e.key === 'v') {
        setTool('select')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [busy, current, selectedId, goNext, goPrev, deleteSelected, confirmCurrent])

  const reviewedPercent = summary?.images_total
    ? Math.round((summary.images_confirmed / summary.images_total) * 100)
    : 0
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <Card
      title="复核 / 修正伪标签"
      size="small"
      extra={
        <Space>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => load(filter, page)} loading={loading}>
            刷新
          </Button>
        </Space>
      }
    >
      <Row gutter={12} style={{ marginBottom: 12 }}>
        <Col xs={12} md={5}>
          <Statistic title="已确认" value={summary?.images_confirmed ?? 0} suffix={`/ ${summary?.images_total ?? 0}`} />
          <Progress percent={reviewedPercent} size="small" />
        </Col>
        <Col xs={12} md={3}>
          <Statistic title="待复核" value={summary?.images_pending ?? 0} />
        </Col>
        <Col xs={12} md={3}>
          <Statistic title="我改过的图" value={summary?.images_touched ?? 0} />
        </Col>
        <Col xs={12} md={3}>
          <Statistic title="人加框" value={summary?.boxes_human_added ?? 0} />
        </Col>
        <Col xs={12} md={3}>
          <Statistic title="改过框" value={summary?.boxes_edited ?? 0} />
        </Col>
        <Col xs={12} md={3}>
          <Statistic title="删过框" value={summary?.boxes_removed ?? 0} />
        </Col>
      </Row>

      {summary?.reviewed ? (
        <Alert type="success" showIcon style={{ marginBottom: 12 }} message="所有图片都已确认" />
      ) : (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="逐张核对：漏检处补框、误检处删框、框位置不对就拖动/缩放。确认后这张图才算复核过。"
        />
      )}

      <Space wrap style={{ marginBottom: 12 }}>
        <Select style={{ width: 130 }} value={filter} onChange={(v) => { setPage(0); setIndex(0); setFilter(v) }} options={FILTERS} />
        <Segmented
          value={tool}
          onChange={(v) => setTool(v as 'select' | 'draw')}
          options={[
            { value: 'select', label: '选择/移动' },
            { value: 'draw', label: '画框 (D)' },
          ]}
        />
        <AutoComplete
          style={{ width: 160 }}
          value={activeCategory}
          onChange={setActiveCategory}
          options={categoryOptions}
          placeholder="新框类别"
        />
        <Button icon={<DeleteOutlined />} danger disabled={!selectedId || busy} onClick={deleteSelected}>
          删除选中框
        </Button>
        <Tooltip title="上一张 (←)">
          <Button icon={<LeftOutlined />} onClick={goPrev} disabled={page === 0 && index === 0} />
        </Tooltip>
        <Text type="secondary">
          {samples.length ? index + 1 : 0} / {samples.length}（本页共 {total}）
        </Text>
        <Tooltip title="下一张 (→)">
          <Button icon={<RightOutlined />} onClick={goNext} disabled={page >= totalPages - 1 && index >= samples.length - 1} />
        </Tooltip>
        <Button
          type="primary"
          icon={current?.review_status === 'confirmed' ? <UndoOutlined /> : <CheckCircleOutlined />}
          loading={busy}
          disabled={!current}
          onClick={() => confirmCurrent(current?.review_status === 'confirmed' ? 'pending' : 'confirmed')}
        >
          {current?.review_status === 'confirmed' ? '取消确认' : '确认本张 (Enter)'}
        </Button>
      </Space>

      {current ? (
        <Row gutter={12}>
          <Col xs={24} lg={17}>
            <BoxEditor
              sample={current}
              categories={declaredCategories}
              activeCategory={activeCategory}
              tool={tool}
              selectedId={selectedId}
              busy={busy}
              onSelect={setSelectedId}
              onAdd={(bbox) => {
                if (!activeCategory.trim()) {
                  message.warning('请先填写新框的类别')
                  return
                }
                void run(() =>
                  api.addPrelabelAnnotation(job.id, { image_uid: current.uid, category: activeCategory.trim(), bbox }),
                )
              }}
              onUpdate={(id, patch) => {
                void run(() =>
                  api.updatePrelabelAnnotation(job.id, id, { bbox: patch.bbox }),
                )
              }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" className="mono">
                {current.rel_path}
              </Text>
              <Text type="secondary" style={{ marginLeft: 8 }}>
                {current.width}×{current.height}
              </Text>
              {current.review_status === 'confirmed' && (
                <Tag color="success" style={{ marginLeft: 8 }}>
                  已确认
                </Tag>
              )}
              {current.review_touched && (
                <Tag color="warning" style={{ marginLeft: 4 }}>
                  有过改动
                </Tag>
              )}
            </div>

            {/* 选中框的属性编辑：改类别 / 手动输入坐标 */}
            {selectedId && current.objects.some((o) => o.box_id === selectedId) && (
              <SelectedBoxPanel
                sample={current}
                selectedId={selectedId}
                categories={declaredCategories}
                busy={busy}
                onUpdate={(patch) => run(() => api.updatePrelabelAnnotation(job.id, selectedId, patch))}
              />
            )}
          </Col>

          <Col xs={24} lg={7}>
            <div className="review-list">
              {samples.map((s, i) => (
                <div
                  key={s.uid}
                  className={`review-thumb${i === index ? ' active' : ''}`}
                  onClick={() => {
                    setIndex(i)
                    setSelectedId(null)
                  }}
                >
                  <img src={imageUrl(s.path)} alt={s.rel_path} />
                  <div className="review-thumb-meta">
                    <span className="review-thumb-name">{s.rel_path}</span>
                    <span>
                      <Tag color={s.review_status === 'confirmed' ? 'success' : 'default'} style={{ marginRight: 4 }}>
                        {s.review_status === 'confirmed' ? '已确认' : '待复核'}
                      </Tag>
                      <Text type="secondary">{s.num_objects} 框</Text>
                    </span>
                  </div>
                </div>
              ))}
              {!samples.length && <Empty description="当前筛选下没有图片" />}
            </div>
            {totalPages > 1 && (
              <Space style={{ marginTop: 8 }}>
                <Button size="small" disabled={page === 0} onClick={() => { setPage((p) => p - 1); setIndex(0) }}>
                  上一页
                </Button>
                <Text type="secondary">
                  {page + 1} / {totalPages}
                </Text>
                <Button size="small" disabled={page >= totalPages - 1} onClick={() => { setPage((p) => p + 1); setIndex(0) }}>
                  下一页
                </Button>
              </Space>
            )}
          </Col>
        </Row>
      ) : (
        <Empty description={loading ? '加载中…' : '当前筛选下没有图片'} />
      )}
    </Card>
  )
}

/** 选中框的属性面板：改类别、手动微调坐标、删除。 */
function SelectedBoxPanel({
  sample,
  selectedId,
  categories,
  busy,
  onUpdate,
}: {
  sample: SampleImage
  selectedId: string
  categories: string[]
  busy: boolean
  onUpdate: (patch: { category?: string; bbox?: number[] }) => Promise<void> | void
}) {
  const obj = sample.objects.find((o) => o.box_id === selectedId)
  const bboxKey = obj?.bbox ? obj.bbox.map((v) => Math.round(v)).join(',') : ''
  const [bboxText, setBboxText] = useState(bboxKey)
  const lastKey = useRef(`${selectedId}|${bboxKey}`)

  useEffect(() => {
    const key = `${selectedId}|${bboxKey}`
    if (lastKey.current !== key) {
      lastKey.current = key
      setBboxText(bboxKey)
    }
  }, [selectedId, bboxKey])

  if (!obj) return null

  const applyBbox = () => {
    const parts = bboxText.split(/[,，\s]+/).map((s) => Number(s)).filter((n) => !Number.isNaN(n))
    if (parts.length !== 4) {
      message.warning('坐标需要 4 个数字：x1,y1,x2,y2')
      return
    }
    void onUpdate({ bbox: parts })
  }

  return (
    <Card size="small" style={{ marginTop: 8 }} title={<><EditOutlined /> 选中框</>}>
      <Space wrap>
        <span>
          类别：
          <Select
            style={{ width: 150 }}
            value={obj.category}
            onChange={(v) => {
              void onUpdate({ category: v })
            }}
            options={(categories.length ? categories : [obj.category]).map((c) => ({ value: c, label: c }))}
            showSearch
          />
        </span>
        {obj.bbox && (
          <span>
            坐标：
            <input
              className="bbox-input"
              value={bboxText}
              onChange={(e) => setBboxText(e.target.value)}
              onBlur={applyBbox}
              onKeyDown={(e) => {
                if (e.key === 'Enter') applyBbox()
              }}
            />
            <Text type="secondary" style={{ marginLeft: 6, fontSize: 12 }}>
              x1,y1,x2,y2（绝对像素）
            </Text>
          </span>
        )}
        {obj.source === 'human_review' && <Tag color="warning">人新加</Tag>}
        {obj.review === 'edited' && <Tag color="processing">已改过</Tag>}
      </Space>
    </Card>
  )
}
