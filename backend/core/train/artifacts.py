"""训练过程图像（M2-06）。

ultralytics 会把过程图直接写在训练目录下，名字固定：
    train_batch*.jpg            增强后的训练批次
    val_batch*_labels.jpg       验证集标注
    val_batch*_pred.jpg         验证集预测（最直观的"学到了没有"）
    results.png / PR_curve.png / confusion_matrix.png ...

这里只负责把文件分类列出；返回相对文件名，由 API 层拼受限的访问地址，
避免把磁盘绝对路径暴露给前端。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _classify(name: str) -> tuple:
    """按文件名归类，返回 (组 id, 中文标签)。顺序即优先级。"""
    low = name.lower()
    if low.startswith("val_batch") and "_pred" in low:
        return "val_pred", "验证集预测"
    if low.startswith("val_batch") and "_labels" in low:
        return "val_labels", "验证集标注"
    if low.startswith("train_batch"):
        return "train_batch", "训练批次（增强后）"
    if low.startswith("labels"):
        return "labels", "数据分布"
    if "confusion_matrix" in low:
        return "matrix", "混淆矩阵"
    if "curve" in low or low.startswith("results"):
        return "curve", "指标曲线"
    return "other", "其它"


# 分组展示顺序
GROUP_ORDER = ("val_pred", "val_labels", "train_batch", "labels", "curve", "matrix", "other")
GROUP_LABELS = {
    "val_pred": "预测示例",
    "val_labels": "标注示例",
    "train_batch": "训练批次（增强后）",
    "labels": "数据分布",
    "curve": "指标曲线",
    "matrix": "混淆矩阵",
    "other": "其它",
}


def list_artifacts(run_dir) -> Dict[str, Any]:
    """列出训练目录下的过程图像，按组归类。

    返回的是相对文件名而非绝对路径。
    """
    base = Path(run_dir)
    images: List[Dict[str, Any]] = []
    if not base.is_dir():
        return {"groups": [], "images": [], "total": 0}

    for p in sorted(base.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        group_id, label = _classify(p.name)
        try:
            stat = p.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        images.append(
            {
                "name": p.name,
                "group": group_id,
                "group_label": label,
                "size": size,
                "mtime": mtime,
            }
        )

    # 按组聚合，保留 GROUP_ORDER 声明的顺序
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    seen: Dict[str, Dict[str, Any]] = {}
    for img in images:
        g = img["group"]
        if g not in seen:
            seen[g] = {"id": g, "label": GROUP_LABELS.get(g, g), "count": 0}
        seen[g]["count"] += 1

    groups = sorted(seen.values(), key=lambda g: order.get(g["id"], 999))
    images.sort(key=lambda i: (order.get(i["group"], 999), i["name"]))
    return {"groups": groups, "images": images, "total": len(images)}


def pick_preview(run_dir) -> List[Dict[str, Any]]:
    """挑出最有观看价值的几张（验证预测优先），供训练详情页顶部展示。"""
    data = list_artifacts(run_dir)
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for img in data["images"]:
        by_group.setdefault(img["group"], []).append(img)

    picked: List[Dict[str, Any]] = []
    for group in ("val_pred", "val_labels", "train_batch"):
        picked.extend(by_group.get(group, [])[:2])
    return picked


def resolve_artifact(run_dir, name: str):
    """把相对文件名解析成磁盘路径，并保证没有越出训练目录。"""
    base = Path(run_dir).resolve()
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return candidate
