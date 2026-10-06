"""M7 预标注：用模型给一批图片自动打框，产出**伪标签**。

## 定位（重要，不要误解成「自动标注」）

模型给出的是**伪标签（pseudo-label）**，不是标注：

* 模型没识别到的东西 → **直接漏掉**，不是「标错」而是「没有这个标注」；
* 模型的错误会被**固化成标注**，拿去训练就是误差放大；
* 因此它**只能作为预标注（pre-annotation）帮人省力气**，绝不能替代人工复核。

本模块的产出物一律带上「来自模型预测」的血缘（权重 / conf / 时间），
`dataset_card.json` 里也会写明，可追溯、不掩盖。

## 与其它模块的关系

    core/infer    提供推理（复用它，不重写）
        ↓
    core/prelabel 把推理结果写进统一 IR（本模块）
        ↓
    taxonomy / clean / split / export   全部复用现成的

也就是说本模块**只负责「推理结果 → IR」这一段**，其余一概不重写。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 导出目录名前缀：凡是由预标注生成的数据集都带它，便于一眼区分
PRELABEL_PREFIX = "prelabel"

from ..infer import (
    InferError,
    InferOptions,
    InferSession,
    InferSpec,
    detect_weights_format,
)
from ..io import is_image, probe_size
from ..ir import (
    KIND_BBOX,
    Annotation,
    BBox,
    Category,
    DatasetBundle,
    ImageRecord,
    make_uid,
)

# 伪标签来源标识：写进 Annotation.meta 与 dataset_card，用于追溯
SOURCE_MODEL = "model_prediction"


@dataclass
class PrelabelConfig:
    """预标注参数。"""

    # 权重 / 类名 / 设备（透传给推理会话）
    weights: str = ""
    classes: List[str] = field(default_factory=list)
    device: str = "cpu"
    imgsz: int = 640
    task: str = "detect"
    fmt: str = ""

    # 单帧推理参数
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300

    # 是否把低于 conf 的候选也留作「待复核」（暂不实现，先留字段说明意图）
    # 目前只保留达到 conf 的框，简单可解释。
    source_id: str = "prelabel"

    def options(self) -> InferOptions:
        return InferOptions(conf=self.conf, iou=self.iou, max_det=self.max_det)


@dataclass
class PrelabelReport:
    """一次预标注的结果报告（可序列化，写进数据集血缘）。"""

    ok: bool = True
    error: str = ""
    weights: str = ""
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "cpu"
    task: str = "detect"
    created_at: str = ""
    # 模型自带的（或显式提供的）类别清单，血缘里要能看出「是谁的预测」
    classes: List[str] = field(default_factory=list)

    # 计数
    images_total: int = 0
    images_with_boxes: int = 0
    images_empty: int = 0
    images_failed: int = 0
    boxes_total: int = 0

    # 逐类实例数
    count_by_category: Dict[str, int] = field(default_factory=dict)
    # 失败明细（路径 -> 原因），便于人工排查而不是静默跳过
    failures: List[Dict[str, str]] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "weights": self.weights,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "task": self.task,
            "created_at": self.created_at,
            "classes": list(self.classes),
            "source": SOURCE_MODEL,
            "images_total": self.images_total,
            "images_with_boxes": self.images_with_boxes,
            "images_empty": self.images_empty,
            "images_failed": self.images_failed,
            "boxes_total": self.boxes_total,
            "count_by_category": self.count_by_category,
            "failures": self.failures[:50],
            "duration_sec": round(self.duration_sec, 3),
            "disclaimer": (
                "这些标注来自模型预测（伪标签），可能漏检与误检，"
                "必须经人工复核后才可作为训练数据。"
            ),
        }

    @property
    def summary(self) -> str:
        unit = "个图像标注" if self.task == "classify" else "个框"
        return (
            f"{self.images_total} 张图 → {self.boxes_total} {unit}"
            f"（有标注 {self.images_with_boxes} / 空 {self.images_empty}"
            + (f" / 失败 {self.images_failed}" if self.images_failed else "")
            + "）"
        )


def list_images(root) -> List[Path]:
    """递归收集目录下的图片。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"图片目录不存在: {root}")
    if root.is_file():
        return [root] if is_image(root) else []
    return sorted(p for p in root.rglob("*") if p.is_file() and is_image(p))


def build_bundle(
    image_paths: List[Path],
    config: PrelabelConfig,
    root: Optional[Path] = None,
) -> DatasetBundle:
    """先把图片装进 IR（此时还没有标注）。

    与接入适配器一样：图像先入 IR，标注随后由推理补上。
    """
    root = Path(root) if root else (image_paths[0].parent if image_paths else Path("."))
    bundle = DatasetBundle(source_id=config.source_id, format_name="prelabel", root=str(root))

    for p in image_paths:
        p = Path(p).resolve()
        try:
            rel = str(p.relative_to(root.resolve()))
        except ValueError:
            rel = p.name
        size = probe_size(p)
        width, height = size if size else (0, 0)
        uid = make_uid(config.source_id, str(root), rel)
        bundle.add_image(
            ImageRecord(
                uid=uid,
                path=str(p),
                rel_path=rel,
                width=width,
                height=height,
                source_id=config.source_id,
            )
        )
    return bundle


def annotate_bundle(
    bundle: DatasetBundle,
    config: PrelabelConfig,
    session: Optional[InferSession] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> PrelabelReport:
    """用模型给 bundle 里的图片打框，结果写进 IR。**就地修改 bundle**。

    返回报告。推理失败的单张图会被记录并跳过，不会中断整批——
    一批几百张图因为其中一张损坏而全废是不可接受的。
    """
    started = time.time()
    report = PrelabelReport(
        weights=config.weights,
        conf=config.conf,
        iou=config.iou,
        imgsz=config.imgsz,
        device=config.device,
        task=config.task,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    images = list(bundle.images.values())
    report.images_total = len(images)
    if not images:
        report.ok = False
        report.error = "没有可标注的图片"
        return report

    owns_session = session is None
    session = session or InferSession()

    try:
        spec = InferSpec(
            weights=config.weights,
            classes=list(config.classes),
            device=config.device,
            imgsz=config.imgsz,
            task=config.task,
            fmt=config.fmt or detect_weights_format(config.weights),
            source="prelabel",
        )
        info = session.load(spec)
        report.task = info.get("task", config.task)
        report.weights = spec.weights

        # 登记类别（即使某类这一批没出现，也保留，避免类别顺序漂移）
        for name in info.get("classes") or []:
            bundle.add_category(Category(name=str(name), original_name=str(name)))
        report.classes = [str(n) for n in (info.get("classes") or [])]

        options = config.options()
        for i, image in enumerate(images, start=1):
            try:
                result = session.infer_path(image.path, options)
            except InferError as exc:
                report.images_failed += 1
                report.failures.append({"path": image.path, "error": str(exc)})
                if on_progress:
                    on_progress(i, len(images), image.rel_path or image.path)
                continue

            # worker 业务错误（如图像不可解码）不能当成「这张图没有目标」——
            # 那会把失败混进「空标注」，与 R-37 的「漏检不可掩盖」同一个道理。
            if not result.ok:
                report.images_failed += 1
                report.failures.append(
                    {"path": image.path, "error": result.error or "推理失败"}
                )
                if on_progress:
                    on_progress(i, len(images), image.rel_path or image.path)
                continue

            added = _add_detections(bundle, image, result, config, report)
            report.boxes_total += added
            if added:
                report.images_with_boxes += 1
            else:
                report.images_empty += 1

            if on_progress:
                on_progress(i, len(images), image.rel_path or image.path)

    except InferError as exc:
        report.ok = False
        report.error = str(exc)
    finally:
        if owns_session:
            session.stop()

    report.duration_sec = time.time() - started
    report.count_by_category = bundle.count_by_category()
    return report


def run_prelabel(
    images_dir,
    config: PrelabelConfig,
    session: Optional[InferSession] = None,
    limit: int = 0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[DatasetBundle, PrelabelReport]:
    """一次完整的预标注：收集图片 → 装进 IR → 推理打标 → 记录血缘。

    这是 API / CLI 用的入口；`build_bundle` 与 `annotate_bundle` 是它的两步。
    `limit > 0` 时只取前 N 张（向导预览用），但仍是同一套结果结构。
    结果写入 `bundle.meta["prelabel"]`，导出时会落进 `dataset_card.json`。
    """
    images = list_images(images_dir)
    if limit and limit > 0:
        images = images[:limit]

    bundle = build_bundle(images, config, root=Path(images_dir))
    report = annotate_bundle(bundle, config, session=session, on_progress=on_progress)
    attach_lineage(bundle, report)
    return bundle, report


def attach_lineage(bundle: DatasetBundle, report: PrelabelReport) -> None:
    """把预标注报告写进 IR 的 meta，供导出时写入 dataset_card 血缘。"""
    bundle.meta["prelabel"] = report.to_dict()


def _slug(text: str, max_len: int = 48) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(text or ""))[:max_len].strip("_")


def export_name(stem: str = "", now: Optional[datetime] = None) -> str:
    """生成带 `prelabel` 前缀的导出目录名（不含路径）。"""
    slug = _slug(stem) or "dataset"
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{PRELABEL_PREFIX}_{slug}_{ts}"


def ensure_prelabel_prefix(name: str) -> str:
    """保证导出目录名带 `prelabel` 前缀；空名走自动生成。"""
    name = str(name or "").strip()
    if not name:
        return export_name()
    return name if name.startswith(PRELABEL_PREFIX) else f"{PRELABEL_PREFIX}_{name}"


def _add_detections(bundle, image, result, config: PrelabelConfig, report: PrelabelReport) -> int:
    """把一帧的检测结果写进 IR。返回加入的框数。"""
    if result.task == "classify":
        # 分类模型：整图一个类别，写成图像级标注
        top1 = result.top1
        if not top1 or not top1.get("class_name"):
            return 0
        bundle.add_annotation(
            Annotation(
                image_uid=image.uid,
                category=str(top1["class_name"]),
                bbox=None,
                kind="image",
                score=top1.get("confidence"),
                meta={"source": SOURCE_MODEL, "weights": config.weights, "conf": config.conf},
            )
        )
        return 1

    added = 0
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        bundle.add_annotation(
            Annotation(
                image_uid=image.uid,
                category=str(det.class_name),
                bbox=BBox(x1, y1, x2, y2),
                kind=KIND_BBOX,
                # 保留模型置信度，供人工复核时按可信度排序
                score=det.confidence,
                meta={
                    "source": SOURCE_MODEL,
                    "weights": config.weights,
                    "conf": config.conf,
                },
            )
        )
        added += 1
    return added


__all__ = [
    "PRELABEL_PREFIX",
    "SOURCE_MODEL",
    "PrelabelConfig",
    "PrelabelReport",
    "annotate_bundle",
    "attach_lineage",
    "build_bundle",
    "ensure_prelabel_prefix",
    "export_name",
    "list_images",
    "run_prelabel",
]
