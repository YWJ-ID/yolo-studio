"""数据集接入接口。

本文件只做「校验参数 -> 调用 core -> 序列化」，不含业务逻辑。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException

from core.analytics import AnalyticsConfig, analyze, write_html_report
from core.clean import CleanConfig, clean, list_rules
from core.export import ExportConfig, ExportError, export_yolo
from core.ingest import available_adapters, detect_format, load_and_merge, load_dataset
from core.ir import DatasetBundle
from core.split import SplitConfig, assign_splits
from core.taxonomy import (
    TaxonomyConfig,
    apply_taxonomy,
    suggest_merges,
    validate_class_names,
)

from ...config import settings
from ...schemas import (
    AdapterInfo,
    AnalyticsRequest,
    AnalyticsResponse,
    BrowseRequest,
    BrowseResponse,
    CleanRequest,
    CleanResponse,
    DatasetVersion,
    ExportRequest,
    ExportResponse,
    ReportRequest,
    ReportResponse,
    ScanRequest,
    ScanResponse,
    SplitRequest,
    SplitResponse,
    TaxonomyRequest,
    TaxonomyResponse,
    TaxonomySuggestResponse,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# 扫描响应里最多返回多少个样本图，避免响应过大
_MAX_SAMPLES = 20
# 自动生成输出目录名时的最大长度
_SLUG_MAX = 48


@router.get("/adapters", response_model=list[AdapterInfo])
def list_adapters() -> list[AdapterInfo]:
    """列出已注册的数据格式适配器。"""
    return [AdapterInfo(**a) for a in available_adapters()]


@router.post("/detect")
def detect(req: ScanRequest) -> dict:
    """只探测格式，不读数据。用于导入向导第一步的即时反馈。"""
    root = Path(req.path)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {req.path}")
    fmt = req.fmt or detect_format(root)
    return {"path": str(root), "detected_format": fmt, "recognized": fmt is not None}


@router.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest) -> ScanResponse:
    """扫描数据目录：识别格式、读取标签、返回统计与样本。

    这是数据模块的第一个可用切片，用于导入向导的「预览」步骤。
    """
    root = Path(req.path)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {req.path}")

    try:
        bundle: DatasetBundle = load_dataset(root, fmt=req.fmt, **_adapter_options(req))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # 适配器内部异常
        raise HTTPException(status_code=500, detail=f"解析失败: {exc}") from exc

    samples = _build_samples(bundle)
    return ScanResponse(
        detected_format=bundle.format_name,
        stats=bundle.stats(),
        warnings=bundle.warnings[:200],
        categories=bundle.category_names(),
        samples=samples,
    )


def _adapter_options(req: ScanRequest) -> dict:
    """只传递用户真正设置过的参数，避免用 None 覆盖适配器的默认值。"""
    options: dict = {}
    if req.source_id:
        options["source_id"] = req.source_id
    if req.group_by:
        options["group_by"] = req.group_by
    if req.level is not None:
        # 空字符串表示「使用全部层级」，转换为 None 传给适配器
        options["level"] = req.level or None
    if req.frames_dir:
        options["frames_dir"] = req.frames_dir
    if req.include_objects:
        options["include_objects"] = True
    if req.images_dir:
        options["images_dir"] = req.images_dir
    if req.voc_one_based:
        options["voc_one_based"] = True
    return options


def _build_sample(bundle: DatasetBundle, image) -> dict:
    """把一张图像及其标注整理成前端预览用的字典。

    图像级标注（分类）没有 bbox，此处 bbox 为 None，由前端按 kind 区别渲染。
    """
    anns = bundle.annotations_of(image.uid)
    return {
        "uid": image.uid,
        "path": image.path,
        "rel_path": image.rel_path,
        "width": image.width,
        "height": image.height,
        "split": image.split,
        "group": image.group,
        "source_id": image.source_id,
        "num_objects": len(anns),
        "objects": [
            {
                "category": a.category,
                "kind": a.kind,
                "bbox": list(a.bbox.as_tuple()) if a.bbox else None,
            }
            for a in anns
        ],
    }


def _build_samples(bundle: DatasetBundle) -> list[dict]:
    """取少量图像 + 其标注，供导入向导的预览步骤使用。"""
    return [_build_sample(bundle, im) for im in list(bundle.images.values())[:_MAX_SAMPLES]]


@router.post("/browse", response_model=BrowseResponse)
def browse(req: BrowseRequest) -> BrowseResponse:
    """数据浏览器：按类别 / 划分 / 来源 / 目标数 / 路径子串筛选，分页返回缩略图数据。

    与 scan 一样只读不写。筛选在内存 IR 上做，适合本机规模的数据集；
    排序固定按相对路径，保证翻页稳定（不会因 dict 顺序变化而重复或漏图）。
    """
    bundle = _load_or_400(req)
    opts = req.browse

    def keep(image) -> bool:
        anns = bundle.annotations_of(image.uid)
        if opts.split and image.split != opts.split:
            return False
        if opts.source_id and image.source_id != opts.source_id:
            return False
        if opts.search and opts.search.lower() not in (image.rel_path or "").lower():
            return False
        if opts.min_objects is not None and len(anns) < opts.min_objects:
            return False
        if opts.max_objects is not None and len(anns) > opts.max_objects:
            return False
        edge = max(image.width or 0, image.height or 0)
        if opts.min_edge is not None and edge < opts.min_edge:
            return False
        if opts.max_edge is not None and edge > opts.max_edge:
            return False
        if opts.kind == "bbox" and not any(a.is_bbox for a in anns):
            return False
        if opts.kind == "image" and not any(a.kind == "image" for a in anns):
            return False
        if opts.category and not any(a.category == opts.category for a in anns):
            return False
        return True

    images = [im for im in bundle.images.values() if keep(im)]
    images.sort(key=lambda im: (im.rel_path or "", im.uid))
    total = len(images)
    page = images[opts.offset : opts.offset + opts.limit]

    # 每个来源的图像数（按当前数据集统计，不随筛选变化，便于用户判断筛选范围）
    per_source: dict = {}
    for im in bundle.images.values():
        per_source[im.source_id] = per_source.get(im.source_id, 0) + 1
    meta_sources = {s.get("source_id"): s for s in (bundle.meta.get("sources") or [])}
    sources = []
    for sid, count in sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0])):
        meta = meta_sources.get(sid) or {}
        sources.append(
            {
                "source_id": sid,
                "format": meta.get("format") or bundle.format_name,
                "root": meta.get("root") or bundle.root,
                "images": count,
            }
        )

    return BrowseResponse(
        total=total,
        offset=opts.offset,
        limit=opts.limit,
        stats=bundle.stats(),
        categories=bundle.category_names(),
        sources=sources,
        images=[_build_sample(bundle, im) for im in page],
    )


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------


@router.get("/clean/rules")
def clean_rules() -> list:
    """列出全部清洗规则，供前端展示与选择性关闭。"""
    return list_rules()


@router.post("/clean", response_model=CleanResponse)
def clean_dataset(req: CleanRequest) -> CleanResponse:
    """检查数据问题；clean.apply=true 时执行处置。

    注意：清洗作用在解析出的 IR 上，**不会修改原始数据文件**。
    """
    bundle = _load_or_400(req)
    report = clean(bundle, _clean_config(req.clean), apply=req.clean.apply)
    return CleanResponse(report=report.to_dict())


def _clean_config(options) -> CleanConfig:
    return CleanConfig(
        verify_readable=options.verify_readable,
        check_exact_duplicates=options.check_exact_duplicates,
        check_near_duplicates=options.check_near_duplicates,
        near_duplicate_distance=options.near_duplicate_distance,
        min_class_instances=options.min_class_instances,
        limit=options.limit,
        disabled_rules=list(options.disabled_rules or []),
    )


# ---------------------------------------------------------------------------
# 类别规范化
# ---------------------------------------------------------------------------


def _taxonomy_config(options) -> TaxonomyConfig:
    return TaxonomyConfig(
        mapping=dict(options.mapping or {}),
        keep_classes=list(options.keep_classes) if options.keep_classes else None,
        drop_classes=list(options.drop_classes or []),
        kind_filter=options.kind_filter or None,
        class_order=list(options.class_order) if options.class_order else None,
        sanitize=options.sanitize,
    )


@router.post("/taxonomy/suggest", response_model=TaxonomySuggestResponse)
def taxonomy_suggest(req: TaxonomyRequest) -> TaxonomySuggestResponse:
    """查看类别清单与疑似同义类别建议（不做任何改动）。"""
    bundle = _load_or_400(req)
    names = bundle.category_names()
    return TaxonomySuggestResponse(
        classes=names,
        counts=bundle.count_by_category(),
        suggestions=suggest_merges(bundle),
        name_issues=validate_class_names(names),
    )


@router.post("/taxonomy", response_model=TaxonomyResponse)
def taxonomy_apply(req: TaxonomyRequest) -> TaxonomyResponse:
    """应用类别规范化并返回报告（不改原始数据，也不落盘）。"""
    bundle = _load_or_400(req)
    try:
        report = apply_taxonomy(bundle, _taxonomy_config(req.taxonomy))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TaxonomyResponse(report=report.to_dict())


# ---------------------------------------------------------------------------
# 统计分析与质量报告
# ---------------------------------------------------------------------------


def _analytics_config(options) -> AnalyticsConfig:
    return AnalyticsConfig(
        sample_size=options.sample_size,
        min_class_instances=options.min_class_instances,
        max_image_sizes=options.max_image_sizes,
    )


@router.post("/analytics", response_model=AnalyticsResponse)
def analytics(req: AnalyticsRequest) -> AnalyticsResponse:
    """统计分析与质量检查（JSON，供前端图表使用）。"""
    bundle = _load_or_400(req)
    report = analyze(bundle, _analytics_config(req.analytics))
    return AnalyticsResponse(report=report.to_dict())


@router.post("/analytics/report", response_model=ReportResponse)
def analytics_report(req: ReportRequest) -> ReportResponse:
    """生成自包含的单文件 HTML 报告，返回其路径与下载地址。"""
    bundle = _load_or_400(req)
    report = analyze(bundle, _analytics_config(req.analytics))

    name = req.out_name or _auto_out_name(req.path)
    out_path = _resolve_dir(settings.reports_dir, name, ".html")

    write_html_report(
        report,
        out_path,
        title=req.title,
        embed_samples=req.embed_samples,
    )
    return ReportResponse(
        path=str(out_path),
        url=f"/api/files/download?path={quote(str(out_path))}",
        size_bytes=out_path.stat().st_size,
    )


def _resolve_dir(base: Path, name: str, suffix: str) -> Path:
    """在指定目录下安全地拼出输出路径（防目录穿越）。"""
    base = base.resolve()
    slug = re.sub(r"[^0-9A-Za-z_\-]+", "_", name)[:_SLUG_MAX].strip("_") or "output"
    out = (base / f"{slug}{suffix}").resolve()
    try:
        out.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法的输出名称") from None
    return out


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def _load_or_400(req) -> DatasetBundle:
    """加载数据集（支持多来源合并），把 core 的异常翻译成 HTTP 错误。

    给了 sources 就合并多个来源，否则用顶层 path 加载单个来源。
    """
    sources = getattr(req, "sources", None)
    if sources:
        specs = []
        for s in sources:
            spec: dict = {"path": s.path}
            if s.fmt:
                spec["fmt"] = s.fmt
            if s.source_id:
                spec["source_id"] = s.source_id
            if s.group_by:
                spec["group_by"] = s.group_by
            if s.level is not None:
                spec["level"] = s.level or None
            if s.frames_dir:
                spec["frames_dir"] = s.frames_dir
            if s.include_objects:
                spec["include_objects"] = True
            specs.append(spec)
        for s in specs:
            if not Path(s["path"]).exists():
                raise HTTPException(status_code=404, detail=f"目录不存在: {s['path']}")
        try:
            return load_and_merge(specs)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"解析失败: {exc}") from exc

    root = Path(req.path)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {req.path}")
    try:
        return load_dataset(root, fmt=req.fmt, **_adapter_options(req))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"解析失败: {exc}") from exc


def _split_config(options) -> SplitConfig:
    ratios = tuple(float(x) for x in options.ratios)
    if len(ratios) != 3:
        raise HTTPException(status_code=400, detail="split.ratios 必须是 3 个数")
    return SplitConfig(
        ratios=ratios,  # type: ignore[arg-type]
        seed=options.seed,
        strategy="stratified" if options.stratified else "random",
        respect_groups=options.respect_groups,
        respect_existing=options.respect_existing,
    )


@router.post("/split", response_model=SplitResponse)
def split_dataset(req: SplitRequest) -> SplitResponse:
    """按配置划分数据集并返回分布报告（不落盘，用于预览）。"""
    bundle = _load_or_400(req)
    report = assign_splits(bundle, _split_config(req.split))
    return SplitResponse(split_report=report.to_dict(), warnings=report.warnings)


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


@router.post("/export", response_model=ExportResponse)
def export_dataset(req: ExportRequest) -> ExportResponse:
    """类别规范化 + 清洗 + 划分 + 导出为 ultralytics 可训练的数据集。"""
    bundle = _load_or_400(req)

    # 顺序：先统一类别空间（含标注形态过滤），再清洗，最后划分。
    # 类别与形态先定下来，清洗的"类别名变体"等检查才有意义。
    if req.taxonomy.enabled:
        tax_report = apply_taxonomy(bundle, _taxonomy_config(req.taxonomy))
        if tax_report.removed_annotations or tax_report.merged:
            bundle.warnings.append(
                f"类别规范化：{len(tax_report.classes_before)} -> "
                f"{len(tax_report.classes_after)} 类，"
                f"移除标注 {tax_report.removed_annotations}，"
                f"删除空标注图 {tax_report.removed_images}"
            )

    if req.clean.enabled:
        report = clean(bundle, _clean_config(req.clean), apply=True)
        if report.findings:
            bundle.warnings.append(
                f"导出前清洗：发现 {len(report.findings)} 条问题，"
                f"删图 {report.images_removed}，删标注 {report.annotations_removed}"
            )

    split_report = None
    if req.split.enabled:
        split_report = assign_splits(bundle, _split_config(req.split)).to_dict()

    out_dir = _resolve_out_dir(req)

    config = ExportConfig(
        task=req.task,
        name_style=req.name_style,
        file_mode=req.file_mode,
        overwrite=req.overwrite,
    )
    if req.class_order:
        config.class_order = req.class_order

    try:
        result = export_yolo(bundle, out_dir, config, split_report=split_report)
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExportResponse(out_dir=result.out_dir, report=result.to_dict())


def _resolve_out_dir(req: ExportRequest) -> Path:
    """确定输出目录，并保证它落在存储目录内（防目录穿越）。"""
    base = settings.datasets_dir.resolve()
    name = req.out_name or _auto_out_name(req.path)
    out = (base / name).resolve()
    try:
        out.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="输出目录必须位于数据集存储目录内") from None
    if out == base:
        raise HTTPException(status_code=400, detail="输出目录名不能为空")
    return out


def _auto_out_name(path: str) -> str:
    stem = Path(path).name or "dataset"
    slug = re.sub(r"[^0-9A-Za-z_\-]+", "_", stem)[:_SLUG_MAX].strip("_") or "dataset"
    return f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# 已生成的数据集版本
# ---------------------------------------------------------------------------


@router.get("/versions", response_model=list[DatasetVersion])
def list_versions() -> list[DatasetVersion]:
    """列出存储目录下已生成的数据集（读取各自的 dataset_card.json）。"""
    base = settings.datasets_dir
    if not base.is_dir():
        return []

    versions: list[DatasetVersion] = []
    for card_path in sorted(base.glob("*/dataset_card.json")):
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        versions.append(
            DatasetVersion(
                name=card.get("name", card_path.parent.name),
                path=str(card_path.parent),
                task=card.get("task", "unknown"),
                created_at=card.get("created_at", ""),
                classes=card.get("classes", []),
                images=card.get("images_exported", {}),
                sources=card.get("sources", []),
                prelabel=bool(card.get("prelabel")),
            )
        )
    return sorted(versions, key=lambda v: v.created_at, reverse=True)


@router.get("/versions/{name}", response_model=dict)
def get_version(name: str) -> dict:
    """读取某个数据集版本的完整血缘卡片。"""
    card_path = (settings.datasets_dir / name / "dataset_card.json").resolve()
    try:
        card_path.relative_to(settings.datasets_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法的数据集名称") from None
    if not card_path.is_file():
        raise HTTPException(status_code=404, detail=f"数据集不存在: {name}")
    return json.loads(card_path.read_text(encoding="utf-8"))
