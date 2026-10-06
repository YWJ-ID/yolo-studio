"""M7 预标注：任务单例与导出复用。

预标注要跑一整批推理，CPU 上可能要几十秒到几分钟，因此**不能同步阻塞 HTTP**：
每个任务在后台线程里跑，自己的 `InferSession`（与 M6 实时验证的会话分开，
避免互相把模型换掉）；前端轮询任务状态。

「推理结果 → IR」在 core/prelabel；这里只负责调度与把 IR 交回现成的
taxonomy / clean / split / export 流水线，不重写任何一段。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.clean import CleanConfig, clean
from core.export import ExportConfig, ExportError, export_yolo
from core.infer import InferSession
from core.ir import DatasetBundle
from core.prelabel import (
    PrelabelConfig,
    annotate_bundle,
    attach_lineage,
    build_bundle,
    ensure_prelabel_prefix,
    list_images,
)
from core.split import SplitConfig, assign_splits
from core.taxonomy import TaxonomyConfig, apply_taxonomy

from .config import settings

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"

_lock = threading.RLock()
_jobs: Dict[str, "PrelabelJob"] = {}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class PrelabelJob:
    """一次预标注任务。bundle 只存在内存里，不序列化。"""

    id: str
    status: str = STATUS_PENDING
    images_dir: str = ""
    weights: str = ""
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "cpu"
    task: str = "detect"
    classes: List[str] = field(default_factory=list)
    max_det: int = 300
    source_id: str = "prelabel"
    limit: int = 0
    created_at: str = ""
    updated_at: str = ""
    error: str = ""
    # {done, total, current}
    progress: Dict[str, Any] = field(default_factory=dict)
    report: Optional[Dict[str, Any]] = None
    export: Optional[Dict[str, Any]] = None

    # 内存态：预标注结果 IR（导出时复用）
    bundle: Optional[DatasetBundle] = field(default=None, repr=False)
    # 后台线程/会话（不序列化）
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _session: Optional[InferSession] = field(default=None, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "images_dir": self.images_dir,
            "weights": self.weights,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "task": self.task,
            "classes": list(self.classes),
            "max_det": self.max_det,
            "source_id": self.source_id,
            "limit": self.limit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "progress": dict(self.progress),
            "report": self.report,
            "export": self.export,
            # 供前端判断能否进入「生成数据集」这一步
            "ready": self.status == STATUS_FINISHED and self.bundle is not None,
        }


def start_job(req) -> PrelabelJob:
    """创建并启动一个预标注任务（后台线程），立即返回任务快照。"""
    images_dir = str(getattr(req, "images_dir", "") or "").strip()
    weights = str(getattr(req, "weights", "") or "").strip()
    if not images_dir:
        raise ValueError("images_dir 不能为空")
    if not weights:
        raise ValueError("weights 不能为空")
    if not Path(images_dir).exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not Path(weights).is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    job = PrelabelJob(
        id=f"pl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        images_dir=images_dir,
        weights=weights,
        conf=float(getattr(req, "conf", 0.25) or 0.25),
        iou=float(getattr(req, "iou", 0.7) or 0.7),
        imgsz=int(getattr(req, "imgsz", 640) or 640),
        device=str(getattr(req, "device", "cpu") or "cpu"),
        task=str(getattr(req, "task", "detect") or "detect"),
        classes=[str(c) for c in (getattr(req, "classes", None) or [])],
        max_det=int(getattr(req, "max_det", 300) or 300),
        source_id=str(getattr(req, "source_id", "prelabel") or "prelabel"),
        limit=int(getattr(req, "limit", 0) or 0),
        created_at=_now(),
        updated_at=_now(),
        progress={"done": 0, "total": 0, "current": ""},
    )
    with _lock:
        _jobs[job.id] = job

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    job._thread = thread
    thread.start()
    return job


def _config_of(job: PrelabelJob) -> PrelabelConfig:
    return PrelabelConfig(
        weights=job.weights,
        classes=list(job.classes),
        device=job.device,
        imgsz=job.imgsz,
        task=job.task,
        conf=job.conf,
        iou=job.iou,
        max_det=job.max_det,
        source_id=job.source_id,
    )


def _run_job(job: PrelabelJob) -> None:
    job.status = STATUS_RUNNING
    job.updated_at = _now()
    try:
        images = list_images(job.images_dir)
        if job.limit > 0:
            images = images[: job.limit]
        total = len(images)
        job.progress = {"done": 0, "total": total, "current": ""}

        bundle = build_bundle(images, _config_of(job), root=Path(job.images_dir))

        def on_progress(done: int, _total: int, current: str) -> None:
            job.progress = {"done": done, "total": total, "current": current}
            job.updated_at = _now()

        session = InferSession(python=str(settings.python_executable))
        job._session = session
        try:
            report = annotate_bundle(bundle, _config_of(job), session=session, on_progress=on_progress)
        finally:
            session.stop()
            job._session = None

        attach_lineage(bundle, report)
        job.bundle = bundle
        job.report = report.to_dict()
        if report.ok:
            job.status = STATUS_FINISHED
        else:
            job.status = STATUS_FAILED
            job.error = report.error or "预标注失败"
    except Exception as exc:
        job.status = STATUS_FAILED
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.updated_at = _now()


def list_jobs() -> List[PrelabelJob]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def get_job(job_id: str) -> Optional[PrelabelJob]:
    with _lock:
        return _jobs.get(job_id)


def remove_job(job_id: str) -> bool:
    with _lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        return False
    session = job._session
    if session is not None:
        try:
            session.stop()
        except Exception:
            pass
    return True


def samples(job_id: str, offset: int = 0, limit: int = 12) -> Dict[str, Any]:
    """返回带预测框的预览样本（供前端叠加显示）。"""
    job = get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if job.bundle is None:
        return {"total": 0, "categories": [], "images": []}

    bundle = job.bundle
    images = list(bundle.images.values())
    total = len(images)
    page = images[offset : offset + max(0, limit)]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "categories": bundle.category_names(),
        "images": [_build_sample(bundle, im) for im in page],
    }


def _build_sample(bundle: DatasetBundle, image) -> Dict[str, Any]:
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


def export_job(job_id: str, req) -> Dict[str, Any]:
    """把预标注结果导出为数据集（复用 taxonomy / clean / split / export）。"""
    job = get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if job.status != STATUS_FINISHED or job.bundle is None:
        raise ValueError("任务尚未完成，无法导出")

    bundle = job.bundle

    # 1) 类别规范化（可选）
    taxonomy = getattr(req, "taxonomy", None)
    if taxonomy is not None and getattr(taxonomy, "enabled", False):
        tax_report = apply_taxonomy(
            bundle,
            TaxonomyConfig(
                mapping=dict(getattr(taxonomy, "mapping", {}) or {}),
                keep_classes=list(taxonomy.keep_classes) if taxonomy.keep_classes else None,
                drop_classes=list(taxonomy.drop_classes or []),
                kind_filter=taxonomy.kind_filter or None,
                class_order=list(taxonomy.class_order) if taxonomy.class_order else None,
                sanitize=bool(taxonomy.sanitize),
            ),
        )
        if tax_report.removed_annotations or tax_report.merged:
            bundle.warnings.append(
                f"类别规范化：{len(tax_report.classes_before)} -> {len(tax_report.classes_after)} 类，"
                f"移除标注 {tax_report.removed_annotations}，删除空标注图 {tax_report.removed_images}"
            )

    # 2) 清洗（可选，导出时执行处置）
    clean_opts = getattr(req, "clean", None)
    if clean_opts is not None and getattr(clean_opts, "enabled", False):
        report = clean(
            bundle,
            CleanConfig(
                verify_readable=clean_opts.verify_readable,
                check_exact_duplicates=clean_opts.check_exact_duplicates,
                check_near_duplicates=clean_opts.check_near_duplicates,
                near_duplicate_distance=clean_opts.near_duplicate_distance,
                min_class_instances=clean_opts.min_class_instances,
                limit=clean_opts.limit,
                disabled_rules=list(clean_opts.disabled_rules or []),
            ),
            apply=True,
        )
        if report.findings:
            bundle.warnings.append(
                f"导出前清洗：发现 {len(report.findings)} 条问题，"
                f"删图 {report.images_removed}，删标注 {report.annotations_removed}"
            )

    # 3) 划分（可选）
    split_opts = getattr(req, "split", None)
    split_report = None
    if split_opts is not None and getattr(split_opts, "enabled", False):
        ratios = tuple(float(x) for x in (split_opts.ratios or [0.8, 0.1, 0.1]))
        split_report = assign_splits(
            bundle,
            SplitConfig(
                ratios=ratios,  # type: ignore[arg-type]
                seed=split_opts.seed,
                strategy="stratified" if split_opts.stratified else "random",
                respect_groups=split_opts.respect_groups,
                respect_existing=split_opts.respect_existing,
            ),
        ).to_dict()

    out_dir = _resolve_out_dir(getattr(req, "out_name", "") or "")

    config = ExportConfig(
        task=str(getattr(req, "task", "auto") or "auto"),
        name_style=str(getattr(req, "name_style", "keep") or "keep"),
        file_mode=str(getattr(req, "file_mode", "copy") or "copy"),
        overwrite=bool(getattr(req, "overwrite", False)),
    )
    class_order = getattr(req, "class_order", None)
    if class_order:
        config.class_order = list(class_order)

    try:
        result = export_yolo(bundle, out_dir, config, split_report=split_report)
    except ExportError as exc:
        raise ValueError(str(exc)) from exc

    payload = {"out_dir": result.out_dir, "report": result.to_dict()}
    job.export = payload
    job.updated_at = _now()
    return payload


def _resolve_out_dir(name: str) -> Path:
    """确定输出目录：强制 prelabel 前缀，并保证落在数据集存储目录内。"""
    base = settings.datasets_dir.resolve()
    safe_name = ensure_prelabel_prefix(name)
    out = (base / safe_name).resolve()
    try:
        out.relative_to(base)
    except ValueError:
        raise ValueError("输出目录必须位于数据集存储目录内") from None
    return out


__all__ = [
    "STATUS_FAILED",
    "STATUS_FINISHED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "PrelabelJob",
    "export_job",
    "get_job",
    "list_jobs",
    "remove_job",
    "samples",
    "start_job",
]
