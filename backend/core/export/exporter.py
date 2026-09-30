"""导出为 ultralytics 可训练的目录结构。

支持两种布局，由数据本身的标注形态决定（也可强制指定）：

**目标检测**
    <out>/
    ├── data.yaml
    ├── images/{train,val,test}/*.jpg
    └── labels/{train,val,test}/*.txt      # class xc yc w h（归一化）

**图像分类**
    <out>/
    ├── data.yaml
    └── {train,val,test}/{class_name}/*.jpg

这是「出口归一化」的唯一点：IR 中的绝对像素坐标只在这里换算为 YOLO 归一化坐标。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..ir import (
    KIND_BBOX,
    KIND_IMAGE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Annotation,
    DatasetBundle,
    ImageRecord,
)

_SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

TASK_AUTO = "auto"
TASK_DETECTION = "detection"
TASK_CLASSIFICATION = "classification"
TASKS = (TASK_AUTO, TASK_DETECTION, TASK_CLASSIFICATION)

NAME_KEEP = "keep"
NAME_SOURCE = "source"
NAME_UID = "uid"

MODE_COPY = "copy"
MODE_HARDLINK = "hardlink"
MODE_SYMLINK = "symlink"

# 分类任务遇到多标签图像时的处理
MULTI_FIRST = "first"
MULTI_SKIP = "skip"
MULTI_DUPLICATE = "duplicate"


class ExportError(RuntimeError):
    """导出失败。"""


@dataclass
class ExportConfig:
    task: str = TASK_AUTO
    name_style: str = NAME_KEEP
    file_mode: str = MODE_COPY
    overwrite: bool = False
    classification_multi_label: str = MULTI_FIRST
    class_order: Optional[Sequence[str]] = None
    include_unlabeled: bool = True   # 检测任务：无标注图作为背景负样本导出
    clamp_boxes: bool = True         # 越界框裁剪到图像范围内


@dataclass
class ExportReport:
    out_dir: str = ""
    task: str = ""
    classes: List[str] = field(default_factory=list)
    images_exported: Dict[str, int] = field(default_factory=dict)
    boxes_exported: int = 0
    images_skipped: int = 0
    images_unassigned_to_train: int = 0
    clamped_boxes: int = 0
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    data_yaml: str = ""
    dataset_card: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "task": self.task,
            "classes": self.classes,
            "images_exported": self.images_exported,
            "boxes_exported": self.boxes_exported,
            "images_skipped": self.images_skipped,
            "images_unassigned_to_train": self.images_unassigned_to_train,
            "clamped_boxes": self.clamped_boxes,
            "skipped_by_reason": self.skipped_by_reason,
            "warnings": self.warnings,
            "data_yaml": self.data_yaml,
            "dataset_card": self.dataset_card,
        }


# ---------------------------------------------------------------------------


def detect_task(bundle: DatasetBundle) -> str:
    """根据标注形态推断任务类型。

    纯检测框 -> detection；纯图像级标签 -> classification；两者混合 -> 报错，
    因为混合数据集无法用单一目录布局表达，必须先拆分或过滤。
    """
    has_bbox = any(a.kind == KIND_BBOX for a in bundle.annotations)
    has_image = any(a.kind == KIND_IMAGE for a in bundle.annotations)
    if has_bbox and has_image:
        raise ExportError(
            "数据集中同时存在检测框与图像级标注，无法用单一布局导出。"
            "请先用 taxonomy 模块过滤/拆分，或显式指定 task。"
        )
    if has_bbox:
        return TASK_DETECTION
    if has_image:
        return TASK_CLASSIFICATION
    raise ExportError("数据集中没有任何标注，无法推断任务类型")


def export_yolo(
    bundle: DatasetBundle,
    out_dir,
    config: Optional[ExportConfig] = None,
    split_report: Optional[Dict[str, Any]] = None,
) -> ExportReport:
    """把 DatasetBundle 落盘为 ultralytics 数据集。"""
    config = config or ExportConfig()
    out = Path(out_dir).resolve()
    report = ExportReport(out_dir=str(out))

    if config.task not in TASKS:
        raise ExportError(f"未知 task: {config.task}，可选 {TASKS}")
    report.task = detect_task(bundle) if config.task == TASK_AUTO else config.task

    if out.exists() and any(out.iterdir()):
        if not config.overwrite:
            raise ExportError(f"输出目录非空: {out}（如需覆盖请设置 overwrite=True）")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    classes = _resolve_classes(bundle, config)
    report.classes = classes
    class_index = {c: i for i, c in enumerate(classes)}

    if report.task == TASK_DETECTION:
        _export_detection(bundle, out, config, class_index, report)
    else:
        _export_classification(bundle, out, config, class_index, report)

    _ensure_split_dirs(out, report.task)
    _warn_empty_splits(report)

    report.data_yaml = str(_write_data_yaml(out, report.task, classes))
    report.dataset_card = str(
        _write_dataset_card(out, bundle, config, report, split_report)
    )
    return report


# ---------------------------------------------------------------------------
# 检测布局
# ---------------------------------------------------------------------------


def _export_detection(
    bundle: DatasetBundle,
    out: Path,
    config: ExportConfig,
    class_index: Dict[str, int],
    report: ExportReport,
) -> None:
    used: Dict[str, set] = {s: set() for s in _SPLIT_ORDER}
    report.images_exported = {s: 0 for s in _SPLIT_ORDER}

    for image, split in _iter_images_with_split(bundle, config, report):
        anns = [a for a in bundle.annotations_of(image.uid) if a.kind == KIND_BBOX]

        if not anns:
            if not config.include_unlabeled:
                _skip(report, "检测任务：无标注图（已按配置跳过）")
                continue
            # 背景图：导出空标签文件，这是合法的负样本
        else:
            unknown = [a.category for a in anns if a.category not in class_index]
            if unknown:
                _skip(report, f"含未登记类别 {sorted(set(unknown))[:3]}")
                continue

        line_text, clamped = _labels_to_yolo(anns, image, class_index, config)
        report.clamped_boxes += clamped

        stem = _target_stem(image, out, split, used, config)
        img_dst = out / "images" / split / f"{stem}{image.ext}"
        lbl_dst = out / "labels" / split / f"{stem}.txt"

        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)

        if not _place_file(Path(image.path), img_dst, config, report):
            _skip(report, f"源图不可读: {image.path}")
            continue
        lbl_dst.write_text(line_text, encoding="utf-8")

        report.images_exported[split] += 1
        report.boxes_exported += len(anns)


# 判定"越界"的像素容差。归一化坐标往返会产生 ~1e-6 级误差，
# 若不设容差，会被误报成"裁剪了越界框"，让用户以为数据有问题。
_CLAMP_EPS = 1e-3


def _labels_to_yolo(
    anns: List[Annotation],
    image: ImageRecord,
    class_index: Dict[str, int],
    config: ExportConfig,
) -> tuple:
    """把绝对像素 xyxy 换算为 YOLO 归一化 `class xc yc w h`。"""
    w_img = float(image.width) or 1.0
    h_img = float(image.height) or 1.0
    lines: List[str] = []
    clamped = 0

    for a in anns:
        b = a.bbox
        if b is None:
            continue
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2

        if config.clamp_boxes:
            cx1, cy1 = max(0.0, min(x1, w_img)), max(0.0, min(y1, h_img))
            cx2, cy2 = max(0.0, min(x2, w_img)), max(0.0, min(y2, h_img))
            if (
                abs(cx1 - x1) > _CLAMP_EPS
                or abs(cy1 - y1) > _CLAMP_EPS
                or abs(cx2 - x2) > _CLAMP_EPS
                or abs(cy2 - y2) > _CLAMP_EPS
            ):
                clamped += 1
            x1, y1, x2, y2 = cx1, cy1, cx2, cy2

        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            clamped += 1
            continue

        xc = (x1 + x2) / 2.0 / w_img
        yc = (y1 + y2) / 2.0 / h_img
        nw = bw / w_img
        nh = bh / h_img

        if config.clamp_boxes:
            xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
            nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)

        lines.append(
            f"{class_index[a.category]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
        )

    return ("\n".join(lines) + ("\n" if lines else "")), clamped


# ---------------------------------------------------------------------------
# 分类布局
# ---------------------------------------------------------------------------


def _export_classification(
    bundle: DatasetBundle,
    out: Path,
    config: ExportConfig,
    class_index: Dict[str, int],
    report: ExportReport,
) -> None:
    used: Dict[str, set] = {s: set() for s in _SPLIT_ORDER}
    report.images_exported = {s: 0 for s in _SPLIT_ORDER}

    for image, split in _iter_images_with_split(bundle, config, report):
        labels = sorted(
            {a.category for a in bundle.annotations_of(image.uid) if a.kind == KIND_IMAGE}
        )
        if not labels:
            _skip(report, "分类任务：无标签图")
            continue

        unknown = [c for c in labels if c not in class_index]
        if unknown:
            _skip(report, f"含未登记类别 {unknown[:3]}")
            continue

        if len(labels) > 1:
            mode = config.classification_multi_label
            if mode == MULTI_SKIP:
                _skip(report, f"多标签图（{len(labels)} 个），按配置跳过")
                continue
            if mode == MULTI_FIRST:
                report.warnings.append(
                    f"多标签图仅保留首个类别: {image.rel_path or image.path} -> {labels[0]}"
                )
                labels = labels[:1]

        # 先算出目标名，多标签复制模式下同名只算一次
        stem = _target_stem(image, out, split, used, config)
        placed = False
        for label in labels:
            dst = out / split / label / f"{stem}{image.ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if _place_file(Path(image.path), dst, config, report):
                placed = True
        if not placed:
            _skip(report, f"源图不可读: {image.path}")
            continue

        report.images_exported[split] += 1


# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------


def _resolve_classes(bundle: DatasetBundle, config: ExportConfig) -> List[str]:
    """确定类别顺序。显式给定顺序时，未出现的类别也保留（避免下标漂移）。"""
    if config.class_order:
        classes = [str(c) for c in config.class_order]
        missing = sorted(set(bundle.category_names()) - set(classes))
        if missing:
            raise ExportError(f"class_order 未覆盖这些类别: {missing}")
        return classes
    return bundle.category_names()


def _iter_images_with_split(
    bundle: DatasetBundle,
    config: ExportConfig,
    report: ExportReport,
):
    """按子集顺序遍历图像；未划分的图像归入 train 并计数。"""
    for split in _SPLIT_ORDER:
        for image in bundle.images.values():
            if image.split == split:
                yield image, split

    for image in bundle.images.values():
        if image.split not in _SPLIT_ORDER:
            report.images_unassigned_to_train += 1
            yield image, SPLIT_TRAIN

    if report.images_unassigned_to_train:
        report.warnings.append(
            f"{report.images_unassigned_to_train} 张图未划分，已归入 train。"
            "建议先执行划分（core.split.assign_splits）"
        )


def _target_stem(
    image: ImageRecord,
    out: Path,
    split: str,
    used: Dict[str, set],
    config: ExportConfig,
) -> str:
    """生成不冲突的目标文件名（不含扩展名）。

    合并多个来源时文件名必然冲突（如都是 `000001.jpg`），因此按需加前缀。
    """
    original = Path(image.path).stem
    source_slug = _slug(image.source_id)

    if config.name_style == NAME_UID:
        candidates = [image.uid]
    elif config.name_style == NAME_SOURCE:
        candidates = [f"{source_slug}_{original}"]
    else:
        candidates = [original, f"{source_slug}_{original}"]

    for name in candidates:
        if name not in used[split]:
            used[split].add(name)
            return name

    name = f"{source_slug}_{original}_{image.uid[:8]}"
    used[split].add(name)
    return name


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text)]
    return "".join(keep).strip("_")[:32] or "src"


def _place_file(src: Path, dst: Path, config: ExportConfig, report: ExportReport) -> bool:
    """按配置放置文件。硬链接/软链接失败时自动回退为复制。"""
    if not src.is_file():
        return False
    if dst.exists():
        return True

    mode = config.file_mode
    try:
        if mode == MODE_HARDLINK:
            os.link(src, dst)
            return True
        if mode == MODE_SYMLINK:
            os.symlink(src, dst)
            return True
    except (OSError, NotImplementedError) as exc:
        report.warnings.append(
            f"{mode} 失败（{exc}），已回退为复制。Windows 上软链接通常需要管理员权限。"
        )
        config.file_mode = MODE_COPY

    shutil.copy2(src, dst)
    return True


def _skip(report: ExportReport, reason: str) -> None:
    report.images_skipped += 1
    key = reason.split(":")[0]
    report.skipped_by_reason[key] = report.skipped_by_reason.get(key, 0) + 1


def _ensure_split_dirs(out: Path, task: str) -> None:
    """预先建好三个子集目录，让目录布局可预期。"""
    for split in _SPLIT_ORDER:
        if task == TASK_DETECTION:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        else:
            (out / split).mkdir(parents=True, exist_ok=True)


def _warn_empty_splits(report: ExportReport) -> None:
    """val/test 为空时明确警告——ultralytics 训练必须要有非空 val。"""
    for split in (SPLIT_VAL, SPLIT_TEST):
        if report.images_exported.get(split, 0) == 0:
            report.warnings.append(
                f"{split} 子集为空，无法用于训练/评估。"
                "常见原因：数据来自单个视频或分组过少，防泄漏划分不允许拆散同组数据。"
            )


def _write_data_yaml(out: Path, task: str, classes: List[str]) -> Path:
    """生成 data.yaml。检测与分类的 train/val/test 指向不同。"""
    import yaml

    if task == TASK_DETECTION:
        payload = {
            "path": str(out),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
        }
    else:
        payload = {
            "path": str(out),
            "train": "train",
            "val": "val",
            "test": "test",
        }

    payload["nc"] = len(classes)
    payload["names"] = {i: c for i, c in enumerate(classes)}

    path = out / "data.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path


def _write_dataset_card(
    out: Path,
    bundle: DatasetBundle,
    config: ExportConfig,
    report: ExportReport,
    split_report: Optional[Dict[str, Any]],
) -> Path:
    """记录数据血缘：从哪些源、用什么规则、什么划分种子生成的。"""
    card = {
        "name": out.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": report.task,
        "export": {
            "file_mode": config.file_mode,
            "name_style": config.name_style,
            "clamp_boxes": config.clamp_boxes,
            "include_unlabeled": config.include_unlabeled,
            "classification_multi_label": config.classification_multi_label,
        },
        "classes": report.classes,
        "num_classes": len(report.classes),
        "sources": [
            {"source_id": bundle.source_id, "format": bundle.format_name, "root": bundle.root}
        ],
        "images_exported": report.images_exported,
        "boxes_exported": report.boxes_exported,
        "images_skipped": report.images_skipped,
        "skipped_by_reason": report.skipped_by_reason,
        "clamped_boxes": report.clamped_boxes,
        "split_report": split_report,
        "bundle_stats": bundle.stats(),
        "warnings": report.warnings,
    }

    path = out / "dataset_card.json"
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
