import axios from 'axios'
import type {
  AdapterInfo,
  AnalyticsReport,
  ArtifactsResponse,
  BaseWeightsResponse,
  BrowseResponse,
  CleanReport,
  CleanRule,
  CompareResponse,
  DatasetVersion,
  DeployFormat,
  DeployJob,
  DeployResult,
  DeployVerifyReport,
  EnvResponse,
  EvalJob,
  EvalResult,
  ExportReport,
  HealthResponse,
  InferFormatsResponse,
  InferFormatInfo,
  InferFrameResult,
  InferSessionState,
  InferWeightItem,
  LogsPage,
  MetricsSeries,
  ModelCard,
  ModelSummary,
  PrelabelJob,
  ResourceSample,
  SampleImage,
  ScanResponse,
  SplitReport,
  SystemConfig,
  TaxonomyReport,
  TaxonomySuggestResponse,
  TrainBackendsResponse,
  TrainJob,
  WeightUploadResponse,
} from '../types'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? '',
  timeout: 120000,
})

// 统一错误信息提取
http.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err?.response?.data?.detail
    if (detail) err.message = typeof detail === 'string' ? detail : JSON.stringify(detail)
    return Promise.reject(err)
  },
)

export const api = {
  health: () => http.get<HealthResponse>('/api/system/health').then((r) => r.data),

  env: () => http.get<EnvResponse>('/api/system/env').then((r) => r.data),

  systemConfig: () => http.get<SystemConfig>('/api/system/config').then((r) => r.data),

  adapters: () => http.get<AdapterInfo[]>('/api/datasets/adapters').then((r) => r.data),

  detect: (path: string, fmt?: string) =>
    http
      .post<{ detected_format: string | null; recognized: boolean }>('/api/datasets/detect', {
        path,
        fmt: fmt || null,
      })
      .then((r) => r.data),

  scan: (payload: {
    path: string
    fmt?: string
    source_id?: string
    group_by?: string
    level?: string
    frames_dir?: string
    include_objects?: boolean
    images_dir?: string
    voc_one_based?: boolean
  }) => http.post<ScanResponse>('/api/datasets/scan', payload).then((r) => r.data),

  split: (payload: unknown) =>
    http
      .post<{ ok: boolean; split_report: SplitReport; warnings: string[] }>(
        '/api/datasets/split',
        payload,
      )
      .then((r) => r.data),

  exportDataset: (payload: unknown) =>
    http
      .post<{ ok: boolean; out_dir: string; report: ExportReport }>('/api/datasets/export', payload)
      .then((r) => r.data),

  versions: () => http.get<DatasetVersion[]>('/api/datasets/versions').then((r) => r.data),

  cleanRules: () => http.get<CleanRule[]>('/api/datasets/clean/rules').then((r) => r.data),

  clean: (payload: unknown) =>
    http.post<{ ok: boolean; report: CleanReport }>('/api/datasets/clean', payload).then((r) => r.data),

  taxonomySuggest: (payload: unknown) =>
    http
      .post<TaxonomySuggestResponse>('/api/datasets/taxonomy/suggest', payload)
      .then((r) => r.data),

  taxonomyApply: (payload: unknown) =>
    http
      .post<{ ok: boolean; report: TaxonomyReport }>('/api/datasets/taxonomy', payload)
      .then((r) => r.data),

  analytics: (payload: unknown) =>
    http.post<{ ok: boolean; report: AnalyticsReport }>('/api/datasets/analytics', payload).then((r) => r.data),

  analyticsReport: (payload: unknown) =>
    http
      .post<{ ok: boolean; path: string; url: string; size_bytes: number }>(
        '/api/datasets/analytics/report',
        payload,
      )
      .then((r) => r.data),

  version: (name: string) =>
    http.get<Record<string, unknown>>(`/api/datasets/versions/${encodeURIComponent(name)}`).then((r) => r.data),

  browse: (payload: unknown) => http.post<BrowseResponse>('/api/datasets/browse', payload).then((r) => r.data),

  // ---------- 训练（M2） ----------
  trainBackends: () => http.get<TrainBackendsResponse>('/api/train/backends').then((r) => r.data),

  trainWeights: () => http.get<BaseWeightsResponse>('/api/train/weights').then((r) => r.data),

  uploadTrainWeight: (file: File, overwrite = false) => {
    const form = new FormData()
    form.append('file', file)
    return http
      .post<WeightUploadResponse>('/api/train/weights', form, {
        params: overwrite ? { overwrite: true } : undefined,
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((r) => r.data)
  },

  trainJobs: () => http.get<{ ok: boolean; jobs: TrainJob[] }>('/api/train/jobs').then((r) => r.data),

  trainJob: (id: string) =>
    http
      .get<{
        ok: boolean
        job: TrainJob
        metrics: MetricsSeries
        resources: ResourceSample
        logs: LogsPage
      }>(`/api/train/jobs/${encodeURIComponent(id)}`)
      .then((r) => r.data),

  startTrain: (payload: unknown) =>
    http.post<{ ok: boolean; job: TrainJob }>('/api/train/jobs', payload).then((r) => r.data),

  stopTrain: (id: string) =>
    http.post<{ ok: boolean; job: TrainJob }>(`/api/train/jobs/${encodeURIComponent(id)}/stop`).then((r) => r.data),

  resumeTrain: (id: string) =>
    http.post<{ ok: boolean; job: TrainJob }>(`/api/train/jobs/${encodeURIComponent(id)}/resume`).then((r) => r.data),

  trainMetrics: (id: string, sinceEpoch?: number) =>
    http
      .get<{ ok: boolean; metrics: MetricsSeries }>(`/api/train/jobs/${encodeURIComponent(id)}/metrics`, {
        params: sinceEpoch != null ? { since_epoch: sinceEpoch } : undefined,
      })
      .then((r) => r.data),

  trainLogs: (id: string, offset = 0, limit = 500) =>
    http
      .get<LogsPage & { ok: boolean }>(`/api/train/jobs/${encodeURIComponent(id)}/logs`, {
        params: { offset, limit },
      })
      .then((r) => r.data),

  trainResources: (id: string) =>
    http
      .get<{ ok: boolean; sample: ResourceSample }>(`/api/train/jobs/${encodeURIComponent(id)}/resources`)
      .then((r) => r.data),

  trainArtifacts: (id: string) =>
    http
      .get<ArtifactsResponse & { ok: boolean }>(`/api/train/jobs/${encodeURIComponent(id)}/artifacts`)
      .then((r) => r.data),

  // ---------- 评估（M3） ----------
  evalJobs: () => http.get<{ ok: boolean; jobs: EvalJob[] }>('/api/eval/jobs').then((r) => r.data),

  evalJob: (id: string) =>
    http
      .get<{ ok: boolean; job: EvalJob; result: EvalResult | null; logs: LogsPage }>(
        `/api/eval/jobs/${encodeURIComponent(id)}`,
      )
      .then((r) => r.data),

  startEval: (payload: unknown) =>
    http.post<{ ok: boolean; job: EvalJob }>('/api/eval/jobs', payload).then((r) => r.data),

  stopEval: (id: string) =>
    http.post<{ ok: boolean; job: EvalJob }>(`/api/eval/jobs/${encodeURIComponent(id)}/stop`).then((r) => r.data),

  evalResult: (id: string) =>
    http
      .get<{ ok: boolean; result: EvalResult | null }>(`/api/eval/jobs/${encodeURIComponent(id)}/result`)
      .then((r) => r.data),

  evalReport: (id: string, payload: unknown) =>
    http
      .post<{ ok: boolean; path: string; url: string; size_bytes: number }>(
        `/api/eval/jobs/${encodeURIComponent(id)}/report`,
        payload,
      )
      .then((r) => r.data),

  evalArtifacts: (id: string) =>
    http
      .get<ArtifactsResponse & { ok: boolean }>(`/api/eval/jobs/${encodeURIComponent(id)}/artifacts`)
      .then((r) => r.data),

  // ---------- 模型库（M3） ----------
  models: () => http.get<{ ok: boolean; models: ModelSummary[] }>('/api/models').then((r) => r.data),

  model: (id: string) =>
    http.get<{ ok: boolean; model: ModelCard }>(`/api/models/${encodeURIComponent(id)}`).then((r) => r.data),

  modelResult: (id: string, split?: string) =>
    http
      .get<{ ok: boolean; result: EvalResult }>(`/api/models/${encodeURIComponent(id)}/result`, {
        params: split ? { split } : undefined,
      })
      .then((r) => r.data),

  registerModel: (jobId: string) =>
    http.post<{ ok: boolean; model: ModelCard }>('/api/models/register', { job_id: jobId }).then((r) => r.data),

  importModel: (payload: {
    weights: string
    data_yaml?: string
    name?: string
    task?: string
    classes?: string[]
    imgsz?: number
    batch?: number
    model_id?: string
  }) => http.post<{ ok: boolean; model: ModelCard }>('/api/models/import', payload).then((r) => r.data),

  evalModel: (id: string, payload: unknown) =>
    http.post<{ ok: boolean; job: EvalJob }>(`/api/models/${encodeURIComponent(id)}/eval`, payload).then((r) => r.data),

  compareModels: (payload: unknown) =>
    http.post<CompareResponse>('/api/models/compare', payload).then((r) => r.data),

  // ---------- 部署导出（M4） ----------
  deployFormats: () =>
    http
      .get<{ ok: boolean; formats: DeployFormat[]; defaults: string[] }>('/api/deploy/formats')
      .then((r) => r.data),

  deployJobs: () => http.get<{ ok: boolean; jobs: DeployJob[] }>('/api/deploy/jobs').then((r) => r.data),

  deployJob: (id: string) =>
    http
      .get<{ ok: boolean; job: DeployJob; result: DeployResult | null; logs: LogsPage }>(
        `/api/deploy/jobs/${encodeURIComponent(id)}`,
      )
      .then((r) => r.data),

  startDeploy: (payload: unknown) =>
    http.post<{ ok: boolean; job: DeployJob }>('/api/deploy/jobs', payload).then((r) => r.data),

  stopDeploy: (id: string) =>
    http.post<{ ok: boolean; job: DeployJob }>(`/api/deploy/jobs/${encodeURIComponent(id)}/stop`).then((r) => r.data),

  deployResult: (id: string) =>
    http
      .get<{ ok: boolean; result: DeployResult | null }>(`/api/deploy/jobs/${encodeURIComponent(id)}/result`)
      .then((r) => r.data),

  verifyDeploy: (id: string) =>
    http
      .get<{ ok: boolean; verify: DeployVerifyReport }>(`/api/deploy/jobs/${encodeURIComponent(id)}/verify`)
      .then((r) => r.data),

  // ---------- 实时验证（M6） ----------
  inferFormats: () => http.get<InferFormatsResponse>('/api/infer/formats').then((r) => r.data),

  inferWeights: () =>
    http.get<{ weights: InferWeightItem[] }>('/api/infer/weights').then((r) => r.data),

  inferSession: () =>
    http.get<{ ok: boolean; session: InferSessionState }>('/api/infer/session').then((r) => r.data),

  loadInferModel: (payload: {
    weights: string
    task?: string
    classes?: string[]
    device?: string
    imgsz?: number
    fmt?: string
  }) =>
    http
      .post<{ ok: boolean; session: InferSessionState }>('/api/infer/models', payload)
      .then((r) => r.data),

  closeInferSession: () => http.post<{ ok: boolean; closed: boolean }>('/api/infer/close').then((r) => r.data),

  inferImage: (payload: {
    path?: string
    image?: string
    conf?: number
    iou?: number
    max_det?: number
  }) =>
    http
      .post<{ ok: boolean; result: InferFrameResult }>('/api/infer/image', payload)
      .then((r) => r.data),

  // ---------- 预标注（M7） ----------
  prelabelFormats: () =>
    http.get<{ ok: boolean; formats: InferFormatInfo[] }>('/api/prelabel/formats').then((r) => r.data),

  prelabelWeights: () =>
    http.get<{ weights: InferWeightItem[] }>('/api/prelabel/weights').then((r) => r.data),

  startPrelabel: (payload: {
    weights: string
    images_dir: string
    classes?: string[]
    task?: string
    device?: string
    imgsz?: number
    conf?: number
    iou?: number
    max_det?: number
    fmt?: string
    source_id?: string
    limit?: number
  }) => http.post<{ ok: boolean; job: PrelabelJob }>('/api/prelabel/jobs', payload).then((r) => r.data),

  prelabelJobs: () =>
    http.get<{ ok: boolean; jobs: PrelabelJob[] }>('/api/prelabel/jobs').then((r) => r.data),

  prelabelJob: (id: string) =>
    http.get<{ ok: boolean; job: PrelabelJob }>(`/api/prelabel/jobs/${encodeURIComponent(id)}`).then((r) => r.data),

  deletePrelabel: (id: string) =>
    http.delete<{ ok: boolean; removed: string }>(`/api/prelabel/jobs/${encodeURIComponent(id)}`).then((r) => r.data),

  prelabelSamples: (id: string, offset = 0, limit = 12) =>
    http
      .get<{ ok: boolean; total: number; categories: string[]; images: SampleImage[] }>(
        `/api/prelabel/jobs/${encodeURIComponent(id)}/samples`,
        { params: { offset, limit } },
      )
      .then((r) => r.data),

  exportPrelabel: (id: string, payload: unknown) =>
    http
      .post<{ ok: boolean; out_dir: string; report: ExportReport; job: PrelabelJob }>(
        `/api/prelabel/jobs/${encodeURIComponent(id)}/export`,
        payload,
      )
      .then((r) => r.data),
}

/** 拼出本地图片预览地址。 */
export const imageUrl = (path: string) => `/api/files/image?path=${encodeURIComponent(path)}`

/** 部署产物下载地址（产物在导出目录内，由后端限制访问）。 */
export const deployDownloadUrl = (deployId: string, fmt: string) =>
  `/api/deploy/jobs/${encodeURIComponent(deployId)}/download?fmt=${encodeURIComponent(fmt)}`

/**
 * 拼出后端 WebSocket 地址。
 *
 * 开发环境由 Vite 代理 /api（vite.config.ts 里需开启 ws），生产环境前后端同源，
 * 因此统一用当前页面的 host，只把协议换成 ws/wss。VITE_API_BASE 指向独立后端时才用它。
 */
export function wsUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE as string | undefined
  if (base) {
    const u = new URL(base, window.location.href)
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${u.origin}${path}`
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}
