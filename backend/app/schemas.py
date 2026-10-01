"""API 请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    """扫描一个数据目录，识别格式并统计。"""

    path: str = Field(..., description="数据目录绝对路径")
    fmt: Optional[str] = Field(None, description="强制指定格式，不传则自动探测")
    source_id: Optional[str] = Field(None, description="来源标识")
    group_by: Optional[str] = Field(
        None,
        description="分组键策略：none | parent | stem | regex:<正则>（正则的第一个捕获组作为组名）",
    )

    # --- OpenLABEL / VCD 专用 ---
    level: Optional[str] = Field(
        None, description="取哪个标注层级作为类别（如 driver_actions）；传空字符串表示使用全部层级"
    )
    frames_dir: Optional[str] = Field(None, description="帧图片目录，默认自动探测")
    include_objects: bool = Field(False, description="是否并入 object 类型的标注")

    # --- COCO / VOC 专用 ---
    images_dir: Optional[str] = Field(None, description="图像所在目录，默认自动推断")
    voc_one_based: bool = Field(
        False,
        description="VOC 坐标是否按 1-based 闭区间处理（原始 devkit）；默认按 LabelImg 的 0-based",
    )


class ScanResponse(BaseModel):
    ok: bool = True
    detected_format: str
    stats: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    samples: List[Dict[str, Any]] = Field(default_factory=list)


class AdapterInfo(BaseModel):
    name: str
    display_name: str
    extensions: str


class BrowseOptions(BaseModel):
    """数据浏览器的筛选与分页参数。"""

    offset: int = Field(0, ge=0)
    limit: int = Field(24, ge=1, le=200)
    category: Optional[str] = Field(None, description="只看包含该类别的图像")
    split: Optional[str] = Field(None, description="只看该划分（train / val / test）")
    kind: Optional[str] = Field(None, description="只看该标注形态：bbox（有检测框）| image（图像级）")
    source_id: Optional[str] = Field(None, description="只看该来源")
    search: Optional[str] = Field(None, description="按相对路径子串过滤")
    min_objects: Optional[int] = Field(None, ge=0, description="目标数下限")
    max_objects: Optional[int] = Field(None, ge=0, description="目标数上限")
    min_edge: Optional[int] = Field(None, ge=0, description="图像长边（max(宽,高)）下限，像素")
    max_edge: Optional[int] = Field(None, ge=0, description="图像长边上限，像素")


class BrowseRequest(ScanRequest):
    sources: Optional[List["SourceSpec"]] = None
    browse: BrowseOptions = Field(default_factory=BrowseOptions)


class BrowseResponse(BaseModel):
    ok: bool = True
    total: int = 0
    offset: int = 0
    limit: int = 0
    stats: Dict[str, Any] = Field(default_factory=dict)
    categories: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)


class SplitOptions(BaseModel):
    """划分参数。"""

    enabled: bool = True
    ratios: List[float] = Field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 42
    stratified: bool = Field(True, description="按类别分层，长尾类优先摊匀")
    respect_groups: bool = Field(True, description="同组（同视频）不可拆散，防泄漏")
    respect_existing: bool = Field(True, description="保留输入里已有的划分")


class SourceSpec(BaseModel):
    """一个数据来源。用于多来源合并。"""

    path: str
    fmt: Optional[str] = None
    source_id: Optional[str] = None
    group_by: Optional[str] = None
    level: Optional[str] = None
    frames_dir: Optional[str] = None
    include_objects: bool = False


class CleanOptions(BaseModel):
    """清洗参数。"""

    enabled: bool = True
    apply: bool = Field(
        False, description="false = 仅报告(dry-run)；true = 执行处置。导出流程中会自动置为 true"
    )
    verify_readable: bool = Field(True, description="图像解码校验（较慢但最可靠）")
    check_exact_duplicates: bool = True
    check_near_duplicates: bool = Field(True, description="pHash 近似重复检测（较慢）")
    near_duplicate_distance: int = Field(6, description="pHash 汉明距离阈值，越小越严格")
    min_class_instances: int = 5
    limit: int = Field(0, description=">0 时只检查前 N 张，用于快速预览")
    disabled_rules: List[str] = Field(default_factory=list)


class CleanRequest(ScanRequest):
    sources: Optional[List[SourceSpec]] = None
    clean: CleanOptions = Field(default_factory=CleanOptions)


class CleanResponse(BaseModel):
    ok: bool = True
    report: Dict[str, Any]


class TaxonomyOptions(BaseModel):
    """类别规范化参数。"""

    enabled: bool = True
    mapping: Dict[str, str] = Field(
        default_factory=dict, description="原名 -> 规范名"
    )
    keep_classes: Optional[List[str]] = Field(None, description="只保留这些类别")
    drop_classes: List[str] = Field(default_factory=list)
    kind_filter: Optional[str] = Field(
        None, description="只保留该标注形态：bbox（检测框）| image（图像级）"
    )
    class_order: Optional[List[str]] = Field(None, description="显式指定类别顺序")
    sanitize: bool = Field(False, description="把类名改造为可安全用作目录名的形式")


class TaxonomySuggestResponse(BaseModel):
    ok: bool = True
    classes: List[str] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    name_issues: List[Dict[str, str]] = Field(default_factory=list)


class TaxonomyRequest(ScanRequest):
    sources: Optional[List[SourceSpec]] = None
    taxonomy: TaxonomyOptions = Field(default_factory=TaxonomyOptions)


class TaxonomyResponse(BaseModel):
    ok: bool = True
    report: Dict[str, Any]


class AnalyticsOptions(BaseModel):
    """统计分析参数。"""

    sample_size: int = Field(24, description="抽样预览张数")
    min_class_instances: int = Field(20, description="类别实例数低于此值则提示样本不足")
    max_image_sizes: int = Field(12, description="图像尺寸列表最多展示几种")


class AnalyticsRequest(ScanRequest):
    sources: Optional[List[SourceSpec]] = None
    analytics: AnalyticsOptions = Field(default_factory=AnalyticsOptions)


class AnalyticsResponse(BaseModel):
    ok: bool = True
    report: Dict[str, Any]


class ReportRequest(AnalyticsRequest):
    out_name: Optional[str] = Field(None, description="报告文件名（不含扩展名），默认按来源生成")
    title: str = "数据集质量报告"
    embed_samples: bool = Field(True, description="HTML 中内嵌缩略图（体积更大但可离线分享）")


class ReportResponse(BaseModel):
    ok: bool = True
    path: str
    url: str
    size_bytes: int


class SplitRequest(ScanRequest):
    split: SplitOptions = Field(default_factory=SplitOptions)
    sources: Optional[List[SourceSpec]] = Field(
        None, description="多来源合并：给定时忽略顶层 path，改为合并这些来源"
    )


class SplitResponse(BaseModel):
    ok: bool = True
    split_report: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


class ExportRequest(ScanRequest):
    split: SplitOptions = Field(default_factory=SplitOptions)
    clean: CleanOptions = Field(default_factory=CleanOptions)
    taxonomy: TaxonomyOptions = Field(default_factory=TaxonomyOptions)
    sources: Optional[List[SourceSpec]] = Field(None, description="多来源合并")
    out_name: Optional[str] = Field(None, description="输出目录名，默认按来源自动生成")
    task: str = Field("auto", description="auto | detection | classification")
    name_style: str = Field("keep", description="keep | source | uid")
    file_mode: str = Field("copy", description="copy | hardlink | symlink")
    overwrite: bool = False
    class_order: Optional[List[str]] = Field(None, description="显式指定类别顺序")


class ExportResponse(BaseModel):
    ok: bool = True
    out_dir: str
    report: Dict[str, Any]


class DatasetVersion(BaseModel):
    """已生成的数据集版本。"""

    name: str
    path: str
    task: str
    created_at: str
    classes: List[str]
    images: Dict[str, int] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 训练（M2）
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """新建训练任务的参数。"""

    data_yaml: str = Field(..., description="数据集 data.yaml 路径（通常来自已生成的数据集版本）")
    weights: str = Field(
        "",
        description="权重：.yaml 结构文件从零训练，.pt 预训练权重；留空按 task 取默认值",
    )
    task: str = Field("detect", description="detect | classify")
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = Field("", description="cpu | cuda:0 | ...；留空用服务端配置（auto 时自动探测）")
    workers: int = Field(0, description="dataloader 进程数；CPU 训练建议 0")
    seed: int = 42
    patience: int = 100
    optimizer: str = "auto"
    lr0: Optional[float] = None
    name: Optional[str] = Field(None, description="任务名/训练目录名，留空自动生成")
    extra: Dict[str, Any] = Field(default_factory=dict, description="透传给 ultralytics 的额外参数")


class TrainJobResponse(BaseModel):
    ok: bool = True
    job: Dict[str, Any]


class TrainJobsResponse(BaseModel):
    ok: bool = True
    jobs: List[Dict[str, Any]] = Field(default_factory=list)


class TrainBackendInfo(BaseModel):
    name: str
    display_name: str
    metrics: List[str] = Field(default_factory=list)


class TrainBackendsResponse(BaseModel):
    ok: bool = True
    backends: List[TrainBackendInfo] = Field(default_factory=list)
    monitor: Dict[str, Any] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 评估（M3-01 / M3-02）
# ---------------------------------------------------------------------------


class EvalRequest(BaseModel):
    """新建评估任务的参数。

    两种用法：
      * 给 `job_id`：自动取该训练任务的 best.pt（退而 last.pt）与 data.yaml；
      * 直接给 `weights` + `data_yaml`：评估任意权重。
    """

    job_id: Optional[str] = Field(None, description="训练任务 id，用于自动解析权重与数据集")
    weights: Optional[str] = Field(None, description="权重文件路径（与 job_id 二选一）")
    data_yaml: Optional[str] = Field(None, description="数据集 data.yaml（与 job_id 二选一）")
    split: str = Field("val", description="train | val | test | auto（auto 按 data.yaml 是否声明 test 决定）")
    task: str = Field("detect", description="detect | classify")
    imgsz: int = 640
    batch: int = 16
    device: str = Field("", description="留空用服务端配置")
    workers: int = 0
    conf: float = 0.001
    iou: float = 0.6
    tag: str = Field("", description="展示用标签，会拼进评估目录名")
    extra: Dict[str, Any] = Field(default_factory=dict)


class EvalJobResponse(BaseModel):
    ok: bool = True
    job: Dict[str, Any]


class EvalJobsResponse(BaseModel):
    ok: bool = True
    jobs: List[Dict[str, Any]] = Field(default_factory=list)


class EvalResultResponse(BaseModel):
    ok: bool = True
    result: Optional[Dict[str, Any]] = None


class EvalReportRequest(BaseModel):
    """生成单文件 HTML 评估报告。"""

    out_name: Optional[str] = Field(None, description="报告文件名（不含扩展名），默认按评估任务生成")
    title: str = "模型评估报告"
    embed_images: bool = Field(True, description="内嵌过程图像（体积更大但可离线分享）")


class EvalReportResponse(BaseModel):
    ok: bool = True
    path: str
    url: str
    size_bytes: int


class EvalCompareResponse(BaseModel):
    """多模型 / 多实验对比（M3-04）。"""

    ok: bool = True
    splits: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    class_table: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 模型库（M3-03 / M3-04）
# ---------------------------------------------------------------------------


class ModelSummary(BaseModel):
    """模型列表项。"""

    model_id: str
    name: str = ""
    task: str = ""
    created_at: str = ""
    updated_at: str = ""
    num_classes: int = 0
    classes: List[str] = Field(default_factory=list)
    dataset_name: str = ""
    job_id: str = ""
    has_best: bool = False
    best: Dict[str, Any] = Field(default_factory=dict)
    eval_splits: List[str] = Field(default_factory=list)
    num_evals: int = 0
    deploy_formats: List[str] = Field(default_factory=list)
    num_deploys: int = 0


class ModelListResponse(BaseModel):
    ok: bool = True
    models: List[ModelSummary] = Field(default_factory=list)


class ModelDetailResponse(BaseModel):
    ok: bool = True
    model: Dict[str, Any]


class ModelCompareRequest(BaseModel):
    """对比若干模型（M3-04）。"""

    model_ids: List[str] = Field(..., description="参与对比的模型 id")
    split: Optional[str] = Field(None, description="只比较该划分的评估结果；留空取每个模型最新一次")
    title: str = "模型对比报告"


class ModelEvalRequest(BaseModel):
    """对某个已注册模型发起评估。"""

    split: str = Field("val", description="train | val | test | auto")
    imgsz: Optional[int] = None
    batch: Optional[int] = None
    device: str = ""
    conf: float = 0.001
    iou: float = 0.6
    tag: str = ""


class ModelRegisterRequest(BaseModel):
    """把一次已完成的训练任务注册为模型。"""

    job_id: str = Field(..., description="训练任务 id（训练目录名）")


# ---------------------------------------------------------------------------
# 部署导出（M4）
# ---------------------------------------------------------------------------


class DeployRequest(BaseModel):
    """新建导出任务的参数。

    两种用法（同评估）：
      * 给 `model_id`：自动取模型库里的 best.pt；
      * 直接给 `weights`：导出任意权重。
    """

    model_id: Optional[str] = Field(None, description="模型库中的模型 id")
    job_id: Optional[str] = Field(None, description="训练任务 id（会自动注册为模型后导出）")
    weights: Optional[str] = Field(None, description="权重文件路径")
    formats: List[str] = Field(
        default_factory=lambda: ["onnx", "torchscript"],
        description="导出格式，可多选；不可用的格式会被拒绝并说明原因",
    )
    task: str = Field("detect", description="detect | classify")
    imgsz: int = 640
    batch: int = 1
    device: str = Field("", description="留空用服务端配置")
    half: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: int = Field(0, description="ONNX opset，0 表示用 ultralytics 默认")
    int8: bool = False
    nms: bool = False
    copy_artifacts: bool = Field(True, description="把产物复制到导出目录")
    tag: str = ""


class DeployJobResponse(BaseModel):
    ok: bool = True
    job: Dict[str, Any]


class DeployJobsResponse(BaseModel):
    ok: bool = True
    jobs: List[Dict[str, Any]] = Field(default_factory=list)


class DeployResultResponse(BaseModel):
    ok: bool = True
    result: Optional[Dict[str, Any]] = None


class DeployVerifyResponse(BaseModel):
    ok: bool = True
    verify: Dict[str, Any]


class DeployFormatInfo(BaseModel):
    name: str
    label: str
    suffix: str = ""
    requires: List[str] = Field(default_factory=list)
    needs_gpu: bool = False
    milestone: str = ""
    note: str = ""
    available: bool = False
    reason: str = ""


class DeployFormatsResponse(BaseModel):
    ok: bool = True
    formats: List[DeployFormatInfo] = Field(default_factory=list)
    defaults: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    storage_dir: str


# ---------------------------------------------------------------------------
# M6 实时验证
# ---------------------------------------------------------------------------


class InferFormatInfo(BaseModel):
    """一种可推理权重格式及其实机可用性。"""

    name: str
    label: str
    suffixes: List[str] = Field(default_factory=list)
    requires: List[str] = Field(default_factory=list)
    task_hint: str = ""
    note: str = ""
    available: bool = True
    reason: str = ""


class InferFormatsResponse(BaseModel):
    formats: List[InferFormatInfo] = Field(default_factory=list)
    active: Optional[Dict[str, Any]] = Field(None, description="当前推理会话状态")


class InferWeightItem(BaseModel):
    """一个可选权重。"""

    value: str = Field(..., description="权重路径")
    label: str
    source: str = Field("model_library", description="model_library | deploy | external")
    model_id: str = ""
    task: str = ""
    classes: List[str] = Field(default_factory=list)
    format: str = ""
    exists: bool = True


class InferWeightsResponse(BaseModel):
    weights: List[InferWeightItem] = Field(default_factory=list)


class InferLoadRequest(BaseModel):
    """加载权重。"""

    weights: str
    name: str = ""
    task: str = Field("", description="detect | classify | segment；空则由模型决定")
    classes: List[str] = Field(
        default_factory=list,
        description="显式类别清单，优先于模型自带的类名；ONNX 未写入类名时必须提供",
    )
    device: str = Field("cpu", description="cpu | 0 | 0,1 ...")
    imgsz: int = 640
    half: bool = False
    fmt: str = Field("", description="格式名；空则按扩展名推断")


class InferSessionResponse(BaseModel):
    ok: bool = True
    session: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class InferImageRequest(BaseModel):
    """单张图片推理。二选一：path（服务器本地）或 image（base64）。"""

    path: str = ""
    image: str = Field("", description="base64 编码的图片（不含 data: 前缀）")
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    only_classes: List[int] = Field(default_factory=list)


class InferImageResponse(BaseModel):
    ok: bool = True
    result: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class InferCloseResponse(BaseModel):
    ok: bool = True
    closed: bool = False


class EnvResponse(BaseModel):
    python: str
    python_executable: str = Field("", description="实际运行后端的解释器路径")
    training_python: str = Field("", description="训练将使用的解释器路径（默认与后端相同）")
    platform: str
    ultralytics: Optional[str] = None
    torch: Optional[str] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    devices: List[str] = Field(default_factory=list)
    train_device: str = "cpu"
