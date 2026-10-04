/** 与后端 core/ir.py 对应的前端类型定义。 */

export interface BBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface SampleObject {
  category: string
  kind: 'bbox' | 'image'
  bbox: [number, number, number, number] | null
}

export interface SampleImage {
  uid: string
  path: string
  rel_path: string
  width: number
  height: number
  split: string | null
  group: string | null
  num_objects: number
  objects: SampleObject[]
}

export interface DatasetStats {
  format: string
  root: string
  /** bbox = 目标检测，image = 图像分类，mixed = 混合，unknown = 无标注 */
  annotation_kind: string
  num_images: number
  num_annotations: number
  num_bbox_annotations: number
  num_image_labels: number
  num_categories: number
  /** 分组数（如视频段数），用于防泄漏划分 */
  num_groups: number
  split_counts: Record<string, number>
  unassigned_images: number
  count_by_category: Record<string, number>
  warnings: number
}

export interface ScanResponse {
  ok: boolean
  detected_format: string
  stats: DatasetStats
  warnings: string[]
  categories: string[]
  samples: SampleImage[]
}

export interface AdapterInfo {
  name: string
  display_name: string
  extensions: string
}

export interface SplitReport {
  units_total: number
  units_grouped: number
  images_total: number
  images_already_split: number
  images_assigned: number
  split_images: Record<string, number>
  split_units: Record<string, number>
  class_by_split: Record<string, Record<string, number>>
  classes_missing_in_split: Record<string, string[]>
  group_conflicts: string[]
  warnings: string[]
}

export interface ExportReport {
  out_dir: string
  task: string
  classes: string[]
  images_exported: Record<string, number>
  boxes_exported: number
  images_skipped: number
  images_unassigned_to_train: number
  clamped_boxes: number
  skipped_by_reason: Record<string, number>
  warnings: string[]
  data_yaml: string
  dataset_card: string
}

export interface DatasetVersion {
  name: string
  path: string
  task: string
  created_at: string
  classes: string[]
  images: Record<string, number>
  sources: { source_id: string; format: string; root: string }[]
  /** 是否由预标注（伪标签）生成，需人工复核 */
  prelabel?: boolean
}

export interface CleanFinding {
  rule: string
  severity: 'error' | 'warning' | 'info'
  message: string
  action: string
  image_uid?: string | null
  image_path?: string | null
  category?: string | null
  detail: Record<string, unknown>
}

export interface CleanReport {
  dry_run: boolean
  images_before: number
  images_after: number
  annotations_before: number
  annotations_after: number
  images_removed: number
  annotations_removed: number
  counts_by_severity: Record<string, number>
  counts_by_rule: Record<string, number>
  actions_applied: Record<string, number>
  hashed_sha1: number
  hashed_phash: number
  near_duplicate_pairs: number
  rules_run: string[]
  skipped_rules: string[]
  duration_sec: number
  warnings: string[]
  findings_total: number
  findings: CleanFinding[]
}

export interface CleanRule {
  id: string
  title: string
  category: string
  description: string
}

export interface TaxonomySuggestion {
  key: string
  names: string[]
  suggested: string
  counts: Record<string, number>
  total: number
}

export interface TaxonomyNameIssue {
  name: string
  level: 'error' | 'warning'
  message: string
}

export interface TaxonomySuggestResponse {
  ok: boolean
  classes: string[]
  counts: Record<string, number>
  suggestions: TaxonomySuggestion[]
  name_issues: TaxonomyNameIssue[]
}

export interface TaxonomyReport {
  classes_before: string[]
  classes_after: string[]
  merged: Record<string, string>
  dropped_classes: string[]
  classes_empty: string[]
  removed_annotations: number
  removed_by_kind: number
  removed_by_class: number
  removed_images: number
  name_issues: TaxonomyNameIssue[]
  warnings: string[]
}

export interface ClassDistributionItem {
  name: string
  count: number
  share: number
}

// ---------- M6 实时验证 ----------

export interface InferFormatInfo {
  name: string
  label: string
  suffixes: string[]
  requires: string[]
  task_hint: string
  note: string
  available: boolean
  reason: string
}

export interface InferSessionState {
  alive: boolean
  loaded: boolean
  task: string
  classes: string[]
  num_classes: number
  weights: string
  format: string
  device: string
  imgsz: number
  last_error: string
  /** 类名来源：request = 显式指定，model = 模型自带，空 = 都没有 */
  class_source?: string
}

export interface InferFormatsResponse {
  formats: InferFormatInfo[]
  active: InferSessionState | null
}

export interface InferWeightItem {
  value: string
  label: string
  source: string
  model_id: string
  task: string
  classes: string[]
  format: string
  exists: boolean
}

export interface InferDetection {
  class_index: number
  class_name: string
  confidence: number | null
  bbox: [number, number, number, number]
  mask_area_ratio: number | null
}

export interface InferFrameResult {
  ok: boolean
  error: string
  frame_id: number | null
  duration_ms: number | null
  task: string
  width: number
  height: number
  detections: InferDetection[]
  top1: { class_index: number; class_name: string; confidence: number | null } | null
  topk: { class_index: number; class_name: string; confidence: number | null }[]
  class_names: string[]
  num_detections: number
}

export interface Histogram {
  labels: string[]
  counts: number[]
  edges: number[]
}

export interface AnalyticsReport {
  summary: {
    root?: string
    num_images: number
    num_annotations: number
    num_bbox_annotations: number
    num_image_labels: number
    annotation_kind: string
    num_classes: number
    num_groups: number
    split_counts: Record<string, number>
    unassigned_images: number
  }
  class_distribution: ClassDistributionItem[]
  class_by_split: Record<string, Record<string, number>>
  size_category: { name: string; count: number }[]
  area_ratio_histogram: Histogram
  objects_per_image_histogram: Histogram
  image_sizes: { size: string; count: number }[]
  split_summary: {
    split: string
    images: number
    annotations: number
    bbox_annotations: number
    classes_present: number
  }[]
  imbalance: {
    max_class?: string
    max_count?: number
    min_class?: string
    min_count?: number
    ratio?: number | null
    top20pct_share?: number
    underrepresented?: { name: string; count: number }[]
  }
  findings: { level: string; message: string }[]
  recommendations: string[]
  samples: SampleImage[]
}

export interface HealthResponse {
  status: string
  app: string
  version: string
  storage_dir: string
}

export interface EnvResponse {
  python: string
  python_executable: string
  training_python: string
  platform: string
  ultralytics: string | null
  torch: string | null
  cuda_available: boolean
  cuda_device_count: number
  devices: string[]
  train_device: string
}

/* ========================= 系统配置（M0-04） ========================= */

export interface ConfigPathInfo {
  key: string
  env: string
  kind: 'dir' | 'file'
  value: string
  exists: boolean
}

export interface OptionalDependency {
  name: string
  purpose: string
  install: string
  installed: boolean
}

export interface SystemConfig {
  ok: boolean
  app: string
  version: string
  platform: string
  api_port: number
  python_executable: string
  training_python: string
  train_device: string
  paths: ConfigPathInfo[]
  allowed_roots: string[]
  files_read_unrestricted: boolean
  cors_origins?: string[]
  frontend_dist: string
  frontend_dist_exists: boolean
  optional_dependencies: OptionalDependency[]
  export_formats: DeployFormat[]
}

/* ============================ 训练（M2） ============================ */

export interface TrainJob {
  id: string
  status: string
  status_label: string
  spec: Record<string, any>
  run_dir: string
  created_at: string
  started_at: string
  ended_at: string
  pid: number | null
  returncode: number | null
  error: string
  message: string
  current_epoch: number
  epochs_total: number
  progress: number
  best: Record<string, any>
  metrics_rows: number
  log_count: number
  attempts: number
  stopped_by_user: boolean
}

export interface MetricRow {
  epoch: number
  values: Record<string, number>
  headline: Record<string, number>
}

export interface MetricsSeries {
  columns: string[]
  epochs: number
  last_epoch: number | null
  rows: MetricRow[]
  best: Record<string, any>
}

export interface GpuDevice {
  index: number
  name?: string
  utilization?: number
  memory_utilization?: number
  mem_used?: number
  mem_total?: number
  mem_percent?: number
  temperature?: number
}

export interface ResourceSample {
  timestamp: number
  cpu: {
    available: boolean
    percent?: number
    count?: number
    mem_percent?: number
    mem_used?: number
    mem_total?: number
    error?: string
    process?: {
      pid: number
      cpu_percent: number
      mem_rss?: number
      mem_percent?: number
      threads?: number
      children?: number
    }
  }
  gpu: { available: boolean; devices: GpuDevice[]; error: string }
}

export interface LogLine {
  seq: number
  text: string
}

export interface LogsPage {
  lines: LogLine[]
  total: number
  next_offset: number
}

export interface ArtifactImage {
  name: string
  group: string
  group_label: string
  size: number
  mtime: number
  url?: string
}

export interface ArtifactGroup {
  id: string
  label: string
  count: number
}

export interface ArtifactsResponse {
  groups: ArtifactGroup[]
  images: ArtifactImage[]
  total: number
  preview?: ArtifactImage[]
}

export interface TrainBackendsResponse {
  ok: boolean
  backends: { name: string; display_name: string; metrics: string[] }[]
  monitor: { cpu_monitor: boolean; gpu_monitor: boolean; gpu_count: number; gpu_error: string }
  defaults: {
    device: string
    python: string
    runs_dir: string
    weights_dir: string
    tasks: string[]
  }
}

/** 一个可作为训练基础模型的权重。 */
export interface BaseWeightItem {
  value: string
  label: string
  /** builtin | weights_dir | model_library | run */
  source: string
  /** 已知时的任务类型；为空表示不限 */
  task: string
  classes: string[]
  /** structure（结构文件，从零训练）| pt（预训练权重） */
  format: string
  exists: boolean
  size_bytes: number
}

export interface BaseWeightsResponse {
  ok: boolean
  weights: BaseWeightItem[]
  weights_dir: string
}

export interface WeightUploadResponse {
  ok: boolean
  name: string
  path: string
  size_bytes: number
  sha256: string
}

/** metrics 事件里的进度是结构体，status 事件里的 progress 是 0~1 的数字。 */
export interface TrainProgress {
  epoch: number
  epochs_done: number
  epochs_total: number
  percent: number
}

/** 训练 WebSocket 事件（snapshot 为快照，其余为增量）。 */
export interface TrainEvent {
  type: 'snapshot' | 'metrics' | 'log' | 'resources' | 'status' | 'finished'
  job_id?: string
  job?: TrainJob
  metrics?: MetricsSeries
  logs?: LogsPage
  resources?: ResourceSample
  monitor?: TrainBackendsResponse['monitor']
  rows?: MetricRow[]
  progress?: number | TrainProgress
  best?: Record<string, any>
  lines?: LogLine[]
  total?: number
  sample?: ResourceSample
  status?: string
  status_label?: string
  message?: string
  error?: string
  current_epoch?: number
  returncode?: number | null
  artifacts?: ArtifactsResponse
}

/* ========================= 评估 / 模型库（M3） ========================= */

export interface ClassMetrics {
  index: number
  name: string
  instances: number
  precision: number
  recall: number
  f1: number
  ap50: number
  ap50_95: number
}

export interface ConfusionMatrix {
  axis: string
  labels: string[]
  matrix: number[][]
}

export interface EvalResult {
  ok: boolean
  error: string
  split: string
  task: string
  weights: string
  data_yaml: string
  model_name: string
  job_id: string
  tag: string
  eval_id: string
  created_at: string
  duration_sec: number
  overall: Record<string, number>
  per_class: ClassMetrics[]
  speed: Record<string, number>
  confusion_matrix: ConfusionMatrix | null
  artifacts: string[]
}

export interface EvalJob {
  id: string
  status: string
  status_label: string
  spec: Record<string, any>
  run_dir: string
  created_at: string
  started_at: string
  ended_at: string
  pid: number | null
  returncode: number | null
  error: string
  message: string
  log_count: number
  stopped_by_user: boolean
  metrics: {
    split?: string
    model_name?: string
    overall?: Record<string, number>
    num_classes?: number
  }
}

export interface ModelSummary {
  model_id: string
  name: string
  task: string
  created_at: string
  updated_at: string
  num_classes: number
  classes: string[]
  dataset_name: string
  job_id: string
  /** training = 训练任务注册；external = 外部权重导入 */
  source?: string
  /** 是否记录了 data.yaml（没有就无法评估） */
  has_data_yaml?: boolean
  has_best: boolean
  best: Record<string, any>
  eval_splits: string[]
  num_evals: number
  deploy_formats: string[]
  num_deploys: number
}

export interface EvalIndexEntry {
  eval_id?: string
  split?: string
  created_at?: string
  run_dir?: string
  overall?: Record<string, number>
  [key: string]: any
}

export interface DeployIndexEntry {
  deploy_id?: string
  created_at?: string
  formats?: string[]
  artifacts?: { format?: string; ok?: boolean; [key: string]: any }[]
  [key: string]: any
}

export interface ModelCard {
  model_id: string
  name: string
  task: string
  created_at: string
  updated_at: string
  classes: string[]
  weights: Record<string, string>
  training: Record<string, any>
  dataset: Record<string, any>
  evals: EvalIndexEntry[]
  deploys: DeployIndexEntry[]
  notes: string[]
}

export interface CompareRow {
  label: string
  model_name: string
  eval_id: string
  split: string
  task: string
  job_id: string
  num_classes: number
  values: Record<string, number | null>
}

export interface CompareResponse {
  ok: boolean
  splits: string[]
  metrics: string[]
  metric_labels?: string[]
  rows: CompareRow[]
  classes: string[]
  class_table: { name: string; values: Record<string, number | null> }[]
  notes: string[]
}

/* ============================ 部署（M4） ============================ */

export interface DeployFormat {
  name: string
  label: string
  suffix: string
  requires: string[]
  needs_gpu: boolean
  milestone: string
  note: string
  available: boolean
  reason: string
}

export interface DeployArtifact {
  format: string
  ok: boolean
  path: string
  source_path: string
  is_dir: boolean
  size: number
  sha256: string
  duration_sec: number
  error: string
  name: string
}

export interface DeployResult {
  ok: boolean
  error: string
  weights: string
  model_name: string
  task: string
  job_id: string
  deploy_id: string
  tag: string
  imgsz: number
  device: string
  requested: string[]
  artifacts: DeployArtifact[]
  created_at: string
  duration_sec: number
}

export interface DeployJob {
  id: string
  status: string
  status_label: string
  spec: Record<string, any>
  run_dir: string
  created_at: string
  started_at: string
  ended_at: string
  pid: number | null
  returncode: number | null
  error: string
  message: string
  log_count: number
  stopped_by_user: boolean
  artifacts: DeployArtifact[]
}

export interface DeployVerifyReport {
  ok: boolean
  checked: number
  problems: { format: string; path: string; issue: string; expected?: number; actual?: number }[]
}

/* ======================= 数据浏览（M5-04） ======================= */

export interface BrowseImage extends SampleImage {
  source_id: string
}

export interface BrowseSource {
  source_id: string
  format: string
  root: string
  images: number
}

export interface BrowseResponse {
  ok: boolean
  total: number
  offset: number
  limit: number
  stats: DatasetStats
  categories: string[]
  sources: BrowseSource[]
  images: BrowseImage[]
}

/* ======================= 预标注（M7）======================== */

export interface PrelabelFailure {
  path: string
  error: string
}

export interface PrelabelReportData {
  ok: boolean
  error: string
  weights: string
  conf: number
  iou: number
  imgsz: number
  device: string
  task: string
  created_at: string
  classes: string[]
  /** 固定为 model_prediction，表示这些标注来自模型预测 */
  source: string
  images_total: number
  images_with_boxes: number
  images_empty: number
  images_failed: number
  boxes_total: number
  count_by_category: Record<string, number>
  failures: PrelabelFailure[]
  duration_sec: number
  /** 免责说明：伪标签必须人工复核 */
  disclaimer: string
}

export interface PrelabelJob {
  id: string
  status: 'pending' | 'running' | 'finished' | 'failed'
  images_dir: string
  weights: string
  conf: number
  iou: number
  imgsz: number
  device: string
  task: string
  classes: string[]
  max_det: number
  source_id: string
  limit: number
  created_at: string
  updated_at: string
  error: string
  progress: { done: number; total: number; current: string }
  report: PrelabelReportData | null
  export: { out_dir: string; report: ExportReport } | null
  /** 是否已完成、可以进入「生成数据集」这一步 */
  ready: boolean
}

