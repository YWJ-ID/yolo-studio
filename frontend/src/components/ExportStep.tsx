import { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { ExportOutlined, PartitionOutlined } from '@ant-design/icons'
import { api } from '../api/client'
import type {
  CleanReport,
  CleanRule,
  ExportReport,
  SplitReport,
  TaxonomyReport,
  TaxonomySuggestResponse,
} from '../types'

const { Text, Paragraph } = Typography

export interface SourceSpec {
  path: string
  fmt?: string
  group_by?: string
  level?: string
  frames_dir?: string
  include_objects?: boolean
  images_dir?: string
  voc_one_based?: boolean
}

interface Props {
  /** 参与合成的一个或多个来源。 */
  sources: SourceSpec[]
  annotationKind: string
  categories: string[]
}

const SEVERITY_META: Record<string, { color: string; label: string }> = {
  error: { color: 'error', label: '严重' },
  warning: { color: 'warning', label: '警告' },
  info: { color: 'default', label: '提示' },
}

const SPLIT_KEYS = ['train', 'val', 'test'] as const

/** 类别清单与同义类名合并建议。 */
function TaxonomyPanel({
  suggest,
  choices,
  onChange,
}: {
  suggest: TaxonomySuggestResponse
  choices: Record<string, string>
  onChange: (next: Record<string, string>) => void
}) {
  const errors = suggest.name_issues.filter((i) => i.level === 'error')
  const warnings = suggest.name_issues.filter((i) => i.level === 'warning')

  return (
    <Card title={`类别体系（${suggest.classes.length} 类）`} size="small" style={{ marginTop: 16 }}>
      <Space wrap>
        {suggest.classes.map((c) => (
          <Tag key={c}>
            {c}
            <Text type="secondary" style={{ marginLeft: 6, fontSize: 12 }}>
              {suggest.counts[c] ?? 0}
            </Text>
          </Tag>
        ))}
      </Space>

      {suggest.suggestions.length > 0 ? (
        <Table
          size="small"
          style={{ marginTop: 12 }}
          rowKey="key"
          pagination={false}
          dataSource={suggest.suggestions}
          columns={[
            {
              title: '疑似同义类名',
              dataIndex: 'names',
              render: (names: string[], r) => (
                <Space wrap>
                  {names.map((n) => (
                    <Tag key={n}>
                      {n}
                      <Text type="secondary" style={{ marginLeft: 4, fontSize: 12 }}>
                        {r.counts[n] ?? 0}
                      </Text>
                    </Tag>
                  ))}
                </Space>
              ),
            },
            { title: '合并后实例数', dataIndex: 'total', width: 120 },
            {
              title: '合并到',
              width: 220,
              render: (_, r) => (
                <Select
                  size="small"
                  style={{ width: 190 }}
                  value={choices[r.key] ?? r.suggested}
                  onChange={(v) => onChange({ ...choices, [r.key]: v })}
                  options={[
                    ...r.names.map((n) => ({ value: n, label: n })),
                    { value: '', label: '不合并' },
                  ]}
                />
              ),
            },
          ]}
        />
      ) : (
        <Alert
          style={{ marginTop: 12 }}
          type="success"
          showIcon
          message="未发现疑似同义类别"
          description="合并建议基于「去掉标点与大小写后是否相同」，只处理写法差异，不猜测拼写错误——错误合并的代价远大于漏合并，拼写差异请人工指定映射。"
        />
      )}

      {errors.length > 0 && (
        <Alert
          style={{ marginTop: 12 }}
          type="error"
          showIcon
          message={`${errors.length} 个类名无法直接落盘`}
          description={
            <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
              {errors.slice(0, 5).map((i, idx) => (
                <li key={idx}>
                  {i.name}: {i.message}
                </li>
              ))}
            </ul>
          }
        />
      )}

      {warnings.length > 0 && (
        <Alert
          style={{ marginTop: 12 }}
          type="warning"
          showIcon
          message="类名提示"
          description={
            <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
              {warnings.slice(0, 5).map((i, idx) => (
                <li key={idx}>
                  {i.name}: {i.message}
                </li>
              ))}
            </ul>
          }
        />
      )}
    </Card>
  )
}

/** 类别规范化预览结果。 */
function TaxonomyResult({ report }: { report: TaxonomyReport }) {
  return (
    <Card title="类别规范化预览（未修改任何数据）" size="small" style={{ marginTop: 16 }}>
      <Descriptions column={1} size="small" bordered>
        <Descriptions.Item label="类别数">
          {report.classes_before.length} → <Text strong>{report.classes_after.length}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="规范化后类别">
          {report.classes_after.length ? (
            report.classes_after.map((c) => <Tag key={c}>{c}</Tag>)
          ) : (
            <Text type="danger">没有剩余类别，数据集将无法训练</Text>
          )}
        </Descriptions.Item>
        {Object.keys(report.merged).length > 0 && (
          <Descriptions.Item label="合并关系">
            {Object.entries(report.merged).map(([from, to]) => (
              <div key={from}>
                <Tag>{from}</Tag> → <Tag color="blue">{to}</Tag>
              </div>
            ))}
          </Descriptions.Item>
        )}
        {report.removed_by_kind > 0 && (
          <Descriptions.Item label="按形态移除标注">
            {report.removed_by_kind} 条
          </Descriptions.Item>
        )}
        {report.removed_by_class > 0 && (
          <Descriptions.Item label="按类别移除标注">
            {report.removed_by_class} 条
          </Descriptions.Item>
        )}
        {report.removed_images > 0 && (
          <Descriptions.Item label="删除的空标注图像">
            {report.removed_images} 张
            <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
              这些图在本形态下已无任何标注
            </Text>
          </Descriptions.Item>
        )}
        {report.classes_empty.length > 0 && (
          <Descriptions.Item label="已无标注的类别">
            {report.classes_empty.map((c) => (
              <Tag key={c}>{c}</Tag>
            ))}
          </Descriptions.Item>
        )}
      </Descriptions>

      {report.warnings.length > 0 && (
        <Alert
          style={{ marginTop: 12 }}
          type="warning"
          showIcon
          message="提示"
          description={
            <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
              {report.warnings.slice(0, 6).map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          }
        />
      )}
    </Card>
  )
}

/** 清洗检查报告。 */
function CleanReportCard({ report, rules }: { report: CleanReport; rules: CleanRule[] }) {
  const ruleTitle = (id: string) => rules.find((r) => r.id === id)?.title ?? id
  const ruleCategory = (id: string) => rules.find((r) => r.id === id)?.category ?? ''

  const rows = Object.entries(report.counts_by_rule).map(([id, count]) => ({
    id,
    title: ruleTitle(id),
    category: ruleCategory(id),
    count,
    severities: [
      ...new Set(
        report.findings.filter((f) => f.rule === id).map((f) => f.severity),
      ),
    ],
  }))

  return (
    <Card title="清洗检查报告（dry-run，未修改任何数据）" size="small" style={{ marginTop: 16 }}>
      <Row gutter={[16, 16]}>
        <Col xs={12} md={4}>
          <Statistic title="问题总数" value={report.findings_total} />
        </Col>
        {(['error', 'warning', 'info'] as const).map((sev) => (
          <Col xs={12} md={4} key={sev}>
            <Statistic
              title={SEVERITY_META[sev].label}
              value={report.counts_by_severity?.[sev] ?? 0}
              valueStyle={{
                color:
                  sev === 'error' && (report.counts_by_severity?.[sev] ?? 0) > 0
                    ? '#cf1322'
                    : undefined,
              }}
            />
          </Col>
        ))}
        <Col xs={12} md={8}>
          <Statistic title="检查耗时" value={report.duration_sec} suffix="s" precision={1} />
        </Col>
      </Row>

      {(report.counts_by_severity?.error ?? 0) > 0 && (
        <Alert
          style={{ marginTop: 12 }}
          type="error"
          showIcon
          message={`发现 ${report.counts_by_severity.error} 个严重问题`}
          description="严重问题多为数据泄漏（同一张图同时出现在 train 与 val/test），会让验证指标虚高。开启「导出前清洗」即可在生成数据集时自动删除。"
        />
      )}

      <Table
        size="small"
        rowKey="id"
        style={{ marginTop: 12 }}
        pagination={false}
        dataSource={rows}
        columns={[
          { title: '类别', dataIndex: 'category', width: 70 },
          { title: '规则', dataIndex: 'id', width: 250 },
          { title: '说明', dataIndex: 'title' },
          { title: '数量', dataIndex: 'count', width: 80 },
          {
            title: '严重度',
            width: 130,
            render: (_, r) =>
              r.severities.map((s: string) => (
                <Tag key={s} color={SEVERITY_META[s]?.color}>
                  {SEVERITY_META[s]?.label ?? s}
                </Tag>
              )),
          },
        ]}
      />

      {report.findings.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ cursor: 'pointer' }}>
            <Text type="secondary">查看前 {Math.min(report.findings.length, 100)} 条明细</Text>
          </summary>
          <Table
            size="small"
            style={{ marginTop: 8 }}
            rowKey={(_, i) => String(i)}
            pagination={{ pageSize: 10, size: 'small' }}
            dataSource={report.findings.slice(0, 100)}
            columns={[
              {
                title: '严重度',
                dataIndex: 'severity',
                width: 80,
                render: (s: string) => (
                  <Tag color={SEVERITY_META[s]?.color}>{SEVERITY_META[s]?.label ?? s}</Tag>
                ),
              },
              { title: '规则', dataIndex: 'rule', width: 200 },
              { title: '问题', dataIndex: 'message' },
              {
                title: '处置',
                dataIndex: 'action',
                width: 130,
                render: (a: string) => (a === 'report' ? <Tag>仅报告</Tag> : <Tag color="blue">{a}</Tag>),
              },
            ]}
          />
        </details>
      )}

      {report.warnings.length > 0 && (
        <Alert
          style={{ marginTop: 12 }}
          type="info"
          showIcon
          message="提示"
          description={
            <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
              {report.warnings.slice(0, 5).map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          }
        />
      )}
    </Card>
  )
}

/** 步骤 2：合并 + 清洗 + 划分 + 导出。 */
export default function ExportStep({ sources, annotationKind, categories }: Props) {
  const [form] = Form.useForm()
  const [splitReport, setSplitReport] = useState<SplitReport | null>(null)
  const [exportReport, setExportReport] = useState<ExportReport | null>(null)
  const [cleanReport, setCleanReport] = useState<CleanReport | null>(null)
  const [rules, setRules] = useState<CleanRule[]>([])
  const [suggest, setSuggest] = useState<TaxonomySuggestResponse | null>(null)
  const [mergeChoices, setMergeChoices] = useState<Record<string, string>>({})
  const [taxReport, setTaxReport] = useState<TaxonomyReport | null>(null)
  const [busy, setBusy] = useState<'' | 'split' | 'export' | 'clean' | 'taxonomy'>('')

  const sourceKey = sources.map((s) => s.path).join('|')

  useEffect(() => {
    api.cleanRules().then(setRules).catch(() => undefined)
  }, [])

  // 来源变化时自动拉取类别清单与同义类名建议
  useEffect(() => {
    setSuggest(null)
    setTaxReport(null)
    const first = sources[0]
    if (!first) return
    api
      .taxonomySuggest({ path: first.path, sources })
      .then((res) => {
        setSuggest(res)
        // 默认采纳系统的合并建议
        const initial: Record<string, string> = {}
        res.suggestions.forEach((s) => {
          initial[s.key] = s.suggested
        })
        setMergeChoices(initial)
      })
      .catch(() => undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey])

  /** 把界面上的合并选择转换成 mapping。 */
  const buildMapping = (): Record<string, string> => {
    const mapping: Record<string, string> = {}
    ;(suggest?.suggestions ?? []).forEach((s) => {
      const target = mergeChoices[s.key]
      if (!target) return
      s.names.forEach((n) => {
        if (n !== target) mapping[n] = target
      })
    })
    return mapping
  }

  const buildTaxonomy = () => {
    const v = form.getFieldsValue()
    const mapping = buildMapping()
    return {
      enabled: true,
      mapping,
      kind_filter: v.tax_kind || null,
      sanitize: !!v.tax_sanitize,
      keep_classes: null,
      drop_classes: [],
      class_order: null,
    }
  }

  const onPreviewTaxonomy = async () => {
    setBusy('taxonomy')
    try {
      const res = await api.taxonomyApply({
        path: sources[0]?.path ?? '',
        sources,
        taxonomy: buildTaxonomy(),
      })
      setTaxReport(res.report)
      message.success(
        `${res.report.classes_before.length} -> ${res.report.classes_after.length} 类`,
      )
    } catch (err: any) {
      message.error(err.message ?? '类别规范化预览失败')
    } finally {
      setBusy('')
    }
  }

  const buildPayload = () => {
    const v = form.getFieldsValue()
    return {
      // 顶层 path 仅为满足接口必填；实际以 sources 为准
      path: sources[0]?.path ?? '',
      sources,
      split: {
        enabled: true,
        ratios: [v.r_train, v.r_val, v.r_test],
        seed: v.seed,
        stratified: v.stratified,
        respect_groups: v.respect_groups,
        respect_existing: v.respect_existing,
      },
      clean: {
        enabled: v.clean_enabled,
        apply: false, // 导出接口内部会自动执行处置，这里只用于预览
        verify_readable: v.clean_verify,
        check_exact_duplicates: true,
        check_near_duplicates: v.clean_near,
        min_class_instances: v.clean_min_class,
        limit: 0,
      },
      taxonomy: buildTaxonomy(),
      out_name: v.out_name || undefined,
      task: v.task,
      file_mode: v.file_mode,
      name_style: 'keep',
      overwrite: !!v.overwrite,
      class_order: categories.length ? categories : undefined,
    }
  }

  const onPreviewClean = async () => {
    setBusy('clean')
    setCleanReport(null)
    try {
      const payload = buildPayload()
      // 预览清洗不应触发导出，这里只发 clean 请求
      const res = await api.clean({
        path: payload.path,
        sources: payload.sources,
        clean: { ...payload.clean, apply: false },
      })
      setCleanReport(res.report)
      message.success(`清洗检查完成，发现 ${res.report.findings_total} 条问题`)
    } catch (err: any) {
      message.error(err.message ?? '清洗检查失败')
    } finally {
      setBusy('')
    }
  }

  const onPreview = async () => {
    setBusy('split')
    setExportReport(null)
    try {
      const res = await api.split(buildPayload())
      setSplitReport(res.split_report)
      message.success('划分预览完成（未落盘）')
    } catch (err: any) {
      message.error(err.message ?? '划分失败')
    } finally {
      setBusy('')
    }
  }

  const onExport = async () => {
    setBusy('export')
    try {
      const res = await api.exportDataset(buildPayload())
      setExportReport(res.report)
      message.success(`已生成数据集: ${res.out_dir}`)
    } catch (err: any) {
      message.error(err.message ?? '导出失败')
    } finally {
      setBusy('')
    }
  }

  const missing = splitReport?.classes_missing_in_split ?? {}
  const missingCount = Object.values(missing).reduce((a, b) => a + b.length, 0)

  return (
    <Card
      title={sources.length > 1 ? `步骤 2：合并 ${sources.length} 个来源 · 划分与导出` : '步骤 2：划分与导出'}
      size="small"
      style={{ marginTop: 16 }}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 12 }}
        message="本步骤会把数据落盘为 ultralytics 可直接训练的数据集"
        description={
          <div>
            {sources.length > 1 && (
              <div>
                将合并以下来源：{sources.map((s, i) => <Tag key={i}>{s.path}</Tag>)}
              </div>
            )}
            {annotationKind === 'image'
              ? '当前是图像级标注，将导出为分类布局 {train,val,test}/{类别}/*.jpg。'
              : annotationKind === 'mixed'
                ? '⚠ 来源中同时存在检测框与图像级标注，属于混合任务，无法用单一目录布局表达，请在下方显式选择任务并对类别做取舍。'
                : '当前带边界框，将导出为检测布局 images/ + labels/ + data.yaml。'}
          </div>
        }
      />

      <Form
        form={form}
        layout="inline"
        initialValues={{
          r_train: 0.8,
          r_val: 0.1,
          r_test: 0.1,
          seed: 42,
          stratified: true,
          respect_groups: true,
          respect_existing: true,
          task: 'auto',
          file_mode: 'copy',
          overwrite: false,
          clean_enabled: true,
          clean_verify: true,
          clean_near: true,
          clean_min_class: 5,
          tax_kind: undefined,
          tax_sanitize: false,
        }}
        style={{ rowGap: 12 }}
      >
        <Form.Item label="比例 train/val/test">
          <Space.Compact>
            <Form.Item name="r_train" noStyle>
              <InputNumber min={0} max={1} step={0.05} style={{ width: 78 }} />
            </Form.Item>
            <Form.Item name="r_val" noStyle>
              <InputNumber min={0} max={1} step={0.05} style={{ width: 78 }} />
            </Form.Item>
            <Form.Item name="r_test" noStyle>
              <InputNumber min={0} max={1} step={0.05} style={{ width: 78 }} />
            </Form.Item>
          </Space.Compact>
        </Form.Item>

        <Form.Item name="seed" label="随机种子">
          <InputNumber style={{ width: 90 }} />
        </Form.Item>

        <Form.Item name="stratified" label="分层" valuePropName="checked" tooltip="按类别分层，稀有类优先摊匀">
          <Switch size="small" />
        </Form.Item>

        <Form.Item
          name="respect_groups"
          label="分组防泄漏"
          valuePropName="checked"
          tooltip="同一视频/序列的帧必须整体落在同一子集，否则指标会虚高"
        >
          <Switch size="small" />
        </Form.Item>

        <Form.Item name="respect_existing" label="保留已有划分" valuePropName="checked">
          <Switch size="small" />
        </Form.Item>

        <Form.Item name="task" label="任务">
          <Select
            style={{ width: 130 }}
            options={[
              { value: 'auto', label: '自动推断' },
              { value: 'detection', label: '目标检测' },
              { value: 'classification', label: '图像分类' },
            ]}
          />
        </Form.Item>

        <Form.Item name="file_mode" label="落盘方式" tooltip="hardlink 不占额外磁盘，但需同盘">
          <Select
            style={{ width: 120 }}
            options={[
              { value: 'copy', label: '复制' },
              { value: 'hardlink', label: '硬链接' },
              { value: 'symlink', label: '软链接' },
            ]}
          />
        </Form.Item>

        <Form.Item name="out_name" label="输出目录名" tooltip="留空则按来源名 + 时间戳自动生成">
          <Input style={{ width: 180 }} placeholder="自动生成" allowClear />
        </Form.Item>

        <Form.Item name="overwrite" label="覆盖同名" valuePropName="checked">
          <Switch size="small" />
        </Form.Item>

        <Form.Item
          name="clean_enabled"
          label="导出前清洗"
          valuePropName="checked"
          tooltip="自动裁剪越界框、删除重复图与跨子集泄漏样本。清洗只作用在解析结果上，不改动原始数据"
        >
          <Switch size="small" />
        </Form.Item>

        <Form.Item name="clean_verify" label="解码校验" valuePropName="checked"
          tooltip="完整解码每张图，可发现损坏文件（较慢）">
          <Switch size="small" />
        </Form.Item>

        <Form.Item name="clean_near" label="近似重复" valuePropName="checked"
          tooltip="pHash 检测裁剪/压缩后的重复（较慢；视频抽帧数据通常很多，属正常）">
          <Switch size="small" />
        </Form.Item>

        <Form.Item name="clean_min_class" label="类别最少实例">
          <InputNumber min={0} style={{ width: 80 }} />
        </Form.Item>

        <Form.Item
          name="tax_kind"
          label="标注形态"
          tooltip="混合来源（检测框 + 图像级）无法用单一目录布局导出，须选定一种形态"
        >
          <Select
            style={{ width: 140 }}
            allowClear
            placeholder="不过滤"
            options={[
              { value: 'bbox', label: '只保留检测框' },
              { value: 'image', label: '只保留图像级' },
            ]}
          />
        </Form.Item>

        <Form.Item name="tax_sanitize" label="类名改造" valuePropName="checked"
          tooltip="把含 / \\ : 等非法字符的类名改造成可安全用作目录名的形式">
          <Switch size="small" />
        </Form.Item>

        <Form.Item>
          <Space>
            <Button icon={<PartitionOutlined />} loading={busy === 'split'} onClick={onPreview}>
              预览划分
            </Button>
            <Button loading={busy === 'taxonomy'} onClick={onPreviewTaxonomy}>
              预览类别
            </Button>
            <Button loading={busy === 'clean'} onClick={onPreviewClean}>
              预览清洗
            </Button>
            <Button
              type="primary"
              icon={<ExportOutlined />}
              loading={busy === 'export'}
              onClick={onExport}
            >
              生成数据集
            </Button>
          </Space>
        </Form.Item>
      </Form>

      {suggest && <TaxonomyPanel suggest={suggest} choices={mergeChoices} onChange={setMergeChoices} />}
      {taxReport && <TaxonomyResult report={taxReport} />}
      {cleanReport && <CleanReportCard report={cleanReport} rules={rules} />}

      {splitReport && (
        <>
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={12} md={5}>
              <Card size="small">
                <Statistic title="划分单元" value={splitReport.units_total} />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  其中分组 {splitReport.units_grouped}
                </Text>
              </Card>
            </Col>
            <Col xs={12} md={5}>
              <Card size="small">
                <Statistic title="图像总数" value={splitReport.images_total} />
              </Card>
            </Col>
            {SPLIT_KEYS.map((k) => (
              <Col xs={12} md={4} key={k}>
                <Card size="small">
                  <Statistic title={k} value={splitReport.split_images[k] ?? 0} />
                </Card>
              </Col>
            ))}
          </Row>

          {splitReport.units_grouped > 1 && (
            <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              已按分组划分：同一组的图像（如同一段视频的帧）不会被拆散到不同子集，
              这是避免验证指标虚高的关键。
            </Paragraph>
          )}

          <Card title="各类别在各子集的分布" size="small" style={{ marginTop: 12 }}>
            <Table
              size="small"
              rowKey="cls"
              pagination={false}
              scroll={{ x: 'max-content' }}
              dataSource={categories.map((cls) => {
                const row: Record<string, unknown> = { cls }
                for (const k of SPLIT_KEYS) row[k] = splitReport.class_by_split?.[k]?.[cls] ?? 0
                return row
              })}
              columns={[
                { title: '类别', dataIndex: 'cls', fixed: 'left' as const },
                ...SPLIT_KEYS.map((k) => ({
                  title: k,
                  dataIndex: k,
                  width: 90,
                  render: (v: number) => (v === 0 ? <Text type="danger">0</Text> : v),
                })),
              ]}
            />
            {missingCount > 0 && (
              <Alert
                style={{ marginTop: 12 }}
                type="warning"
                showIcon
                message={`有 ${missingCount} 处「某类别在某子集缺失」`}
                description={
                  <div>
                    {Object.entries(missing).map(([split, list]) => (
                      <div key={split}>
                        <Tag color="orange">{split}</Tag>
                        {list.join(', ')}
                      </div>
                    ))}
                    <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                      这些类别在该子集上的指标会是 NaN。小样本下「比例精确」与「每类都进各子集」无法兼得，
                      如需强制覆盖请调大固定比例或增加数据。
                    </Paragraph>
                  </div>
                }
              />
            )}
          </Card>

          {splitReport.warnings.filter((w) => !w.includes('缺少')).length > 0 && (
            <Alert
              style={{ marginTop: 12 }}
              type="warning"
              showIcon
              message="划分提示"
              description={
                <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
                  {splitReport.warnings
                    .filter((w) => !w.includes('缺少'))
                    .slice(0, 6)
                    .map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                </ul>
              }
            />
          )}
        </>
      )}

      {exportReport && (
        <Card title="导出结果" size="small" style={{ marginTop: 16 }}>
          <Descriptions column={1} size="small" bordered>
            <Descriptions.Item label="输出目录">
              <Text copyable className="mono">
                {exportReport.out_dir}
              </Text>
            </Descriptions.Item>
            <Descriptions.Item label="任务类型">
              <Tag color={exportReport.task === 'classification' ? 'purple' : 'blue'}>
                {exportReport.task}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="类别">{exportReport.classes.join('、')}</Descriptions.Item>
            <Descriptions.Item label="各子集图像数">
              {SPLIT_KEYS.map((k) => (
                <Tag key={k}>
                  {k}: {exportReport.images_exported[k] ?? 0}
                </Tag>
              ))}
            </Descriptions.Item>
            <Descriptions.Item label="边界框总数">{exportReport.boxes_exported}</Descriptions.Item>
            <Descriptions.Item label="data.yaml">
              <Text copyable className="mono">
                {exportReport.data_yaml}
              </Text>
            </Descriptions.Item>
          </Descriptions>
          {exportReport.warnings.length > 0 && (
            <Alert
              style={{ marginTop: 12 }}
              type="warning"
              showIcon
              message={`${exportReport.warnings.length} 条导出警告`}
              description={
                <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
                  {exportReport.warnings.slice(0, 8).map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              }
            />
          )}
        </Card>
      )}
    </Card>
  )
}
