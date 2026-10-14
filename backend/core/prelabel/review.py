"""M7 复核 / 编辑：人工修正伪标签。

## 定位

预标注产出的是**伪标签**（R-37）：模型漏检 = 缺标注、模型误检 = 错标注。
本模块让人**逐张复核并修正**：

* 增删标注、拖动/缩放框、改类别；
* 把某张图标记为「已确认」或「待复核」。

## 约束

* 编辑只作用于内存里的 `DatasetBundle`，**不碰原始图片文件**。
* 坐标一律**绝对像素 xyxy**（IR 不变量）；显示坐标的换算由前端完成。
* **不掩盖血缘**（与 R-37 同源）：
    - 模型框保留 `meta.source == model_prediction`；
    - 人改过的框记 `meta.review == edited`，并留下 `meta.model_bbox`（原框）；
    - 人新加的框记 `meta.source == human_review`；
    - 删除数（框已不存在，无法从当前 IR 反推）记进 `bundle.meta["review_stats"]`。
  汇总写回 `bundle.meta["prelabel"]["review"]`，导出时落进 `dataset_card.json`，
  让「哪些是模型给的、哪些是人改的」一目了然。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..ir import (
    KIND_BBOX,
    KIND_IMAGE,
    Annotation,
    BBox,
    DatasetBundle,
    ImageRecord,
    make_uid,
)

# 人新加的标注的来源标识
SOURCE_HUMAN = "human_review"

# 复核状态（写在 ImageRecord.meta["review_status"]）
REVIEW_PENDING = "pending"
REVIEW_CONFIRMED = "confirmed"
VALID_REVIEW_STATUS = (REVIEW_PENDING, REVIEW_CONFIRMED)

# 复核动作（写在 Annotation.meta["review"]）
ACTION_ADDED = "added"
ACTION_EDITED = "edited"

# 内部计数键（删除数无法从当前 IR 反推，必须累计）
_STATS_KEY = "review_stats"
_SEQ_KEY = "_box_seq"


class ReviewError(ValueError):
    """复核操作的输入不合法（API 层转成 400）。"""


def next_annotation_id(bundle: DatasetBundle) -> str:
    """给一条标注分配稳定 id。

    标注的「身份」跨请求不能漂移（前端按 id 增删改），因此不用列表下标。
    计数存在 bundle.meta 里，保证同一 bundle 内唯一。
    """
    seq = int(bundle.meta.get(_SEQ_KEY, 0)) + 1
    bundle.meta[_SEQ_KEY] = seq
    return make_uid("ann", str(seq))


def _stats(bundle: DatasetBundle) -> Dict[str, int]:
    stats = bundle.meta.get(_STATS_KEY)
    if not isinstance(stats, dict):
        stats = {"boxes_removed": 0}
        bundle.meta[_STATS_KEY] = stats
    stats.setdefault("boxes_removed", 0)
    return stats


def _touch(bundle: DatasetBundle) -> None:
    """任何一次真正的改动后刷新血缘里的复核汇总。"""
    attach_review_summary(bundle)


def find_annotation(
    bundle: DatasetBundle, ann_id: str
) -> Optional[Tuple[ImageRecord, Annotation]]:
    """按标注 id 找到 (图像, 标注)；找不到返回 None。"""
    if not ann_id:
        return None
    for ann in bundle.annotations:
        if ann.meta.get("box_id") == ann_id:
            image = bundle.images.get(ann.image_uid)
            if image is None:
                return None
            return image, ann
    return None


def _validate_category(category: Any) -> str:
    name = str(category or "").strip()
    if not name:
        raise ReviewError("类别不能为空")
    return name


def _validate_bbox(image: ImageRecord, bbox: Any) -> BBox:
    """校验并归一化一个框。

    人的框不可能落在图像之外，因此显式夹取到图像边界——这是编辑动作本身的
    自然语义，不是对源数据的静默裁剪。宽高必须大于 0，否则无从导出。
    """
    if bbox is None or len(bbox) != 4:
        raise ReviewError("bbox 需要 4 个数 [x1, y1, x2, y2]")
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise ReviewError("bbox 必须是数字") from None

    width = float(image.width or 0)
    height = float(image.height or 0)
    if width > 0 and height > 0:
        x1 = min(max(x1, 0.0), width)
        x2 = min(max(x2, 0.0), width)
        y1 = min(max(y1, 0.0), height)
        y2 = min(max(y2, 0.0), height)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ReviewError("框的宽高必须大于 0")
    return BBox(x1, y1, x2, y2)


def add_annotation(
    bundle: DatasetBundle,
    image_uid: str,
    category: Any,
    bbox: Any = None,
    kind: str = KIND_BBOX,
) -> Annotation:
    """新增一条人工标注（检测框或图像级标签）。"""
    image = bundle.images.get(image_uid)
    if image is None:
        raise ReviewError(f"图像不存在: {image_uid}")
    name = _validate_category(category)

    if kind == KIND_IMAGE:
        box = None
    elif kind == KIND_BBOX:
        box = _validate_bbox(image, bbox)
    else:
        raise ReviewError(f"不支持的标注形态: {kind}")

    ann = Annotation(
        image_uid=image_uid,
        category=name,
        bbox=box,
        kind=kind,
        score=None,
        meta={
            "source": SOURCE_HUMAN,
            "review": ACTION_ADDED,
            "box_id": next_annotation_id(bundle),
        },
    )
    bundle.add_annotation(ann)
    image.meta["review_touched"] = True
    _touch(bundle)
    return ann


def update_annotation(
    bundle: DatasetBundle,
    ann_id: str,
    category: Optional[Any] = None,
    bbox: Any = None,
) -> Tuple[ImageRecord, Annotation]:
    """修改一条标注的坐标 / 类别。返回 (图像, 标注)。"""
    found = find_annotation(bundle, ann_id)
    if found is None:
        raise ReviewError(f"标注不存在: {ann_id}")
    image, ann = found

    changed = False
    if bbox is not None:
        if ann.kind == KIND_IMAGE:
            raise ReviewError("图像级标注没有坐标可改")
        new_box = _validate_bbox(image, bbox)
        if ann.bbox is None or ann.bbox.as_tuple() != new_box.as_tuple():
            # 第一次编辑模型框时留下原坐标，便于追溯「模型原本标在哪」
            if "model_bbox" not in ann.meta and ann.meta.get("source") != SOURCE_HUMAN:
                ann.meta["model_bbox"] = ann.bbox.to_dict() if ann.bbox else None
            ann.bbox = new_box
            changed = True

    if category is not None:
        name = _validate_category(category)
        if name != ann.category:
            ann.category = name
            changed = True

    if changed:
        ann.meta["review"] = ACTION_EDITED
        image.meta["review_touched"] = True
        _touch(bundle)
    return image, ann


def remove_annotation(bundle: DatasetBundle, ann_id: str) -> ImageRecord:
    """删除一条标注，返回它所属的图像。删除数会累计进血缘。"""
    found = find_annotation(bundle, ann_id)
    if found is None:
        raise ReviewError(f"标注不存在: {ann_id}")
    image, ann = found
    if not bundle.remove_annotation(ann):
        raise ReviewError(f"标注不存在: {ann_id}")
    image.meta["review_touched"] = True
    _stats(bundle)["boxes_removed"] += 1
    _touch(bundle)
    return image


def set_image_status(bundle: DatasetBundle, image_uid: str, status: Any) -> ImageRecord:
    """标记某张图的复核状态。"""
    image = bundle.images.get(image_uid)
    if image is None:
        raise ReviewError(f"图像不存在: {image_uid}")
    value = str(status or "").strip().lower()
    if value not in VALID_REVIEW_STATUS:
        raise ReviewError(f"复核状态只能是 {VALID_REVIEW_STATUS}")
    image.meta["review_status"] = value
    _touch(bundle)
    return image


def image_review_status(image: ImageRecord) -> str:
    """图像复核状态（未标记视为待复核）。"""
    return image.meta.get("review_status") or REVIEW_PENDING


def review_summary(bundle: DatasetBundle) -> Dict[str, Any]:
    """当前复核进度汇总（可序列化，写进血缘）。"""
    images = list(bundle.images.values())
    total = len(images)
    confirmed = sum(1 for im in images if image_review_status(im) == REVIEW_CONFIRMED)
    touched = sum(1 for im in images if im.meta.get("review_touched"))

    annotations = bundle.annotations
    model = sum(1 for a in annotations if a.meta.get("source") != SOURCE_HUMAN)
    human_added = sum(1 for a in annotations if a.meta.get("source") == SOURCE_HUMAN)
    edited = sum(1 for a in annotations if a.meta.get("review") == ACTION_EDITED)

    return {
        "images_total": total,
        "images_confirmed": confirmed,
        "images_pending": total - confirmed,
        "images_touched": touched,
        "boxes_total": len(annotations),
        "boxes_model": model,
        "boxes_human_added": human_added,
        "boxes_edited": edited,
        "boxes_removed": int(_stats(bundle).get("boxes_removed", 0)),
        "reviewed": total > 0 and confirmed == total,
    }


def attach_review_summary(bundle: DatasetBundle) -> Optional[Dict[str, Any]]:
    """把复核汇总写进 `bundle.meta["prelabel"]["review"]`（非预标注数据集不动）。"""
    prelabel = bundle.meta.get("prelabel")
    if not isinstance(prelabel, dict):
        return None
    summary = review_summary(bundle)
    prelabel["review"] = summary
    return summary


def reset_summary(bundle: DatasetBundle) -> None:
    """丢弃累计计数（测试/重跑用）。"""
    bundle.meta.pop(_STATS_KEY, None)
    attach_review_summary(bundle)


__all__ = [
    "ACTION_ADDED",
    "ACTION_EDITED",
    "REVIEW_CONFIRMED",
    "REVIEW_PENDING",
    "SOURCE_HUMAN",
    "VALID_REVIEW_STATUS",
    "ReviewError",
    "add_annotation",
    "attach_review_summary",
    "find_annotation",
    "image_review_status",
    "next_annotation_id",
    "remove_annotation",
    "reset_summary",
    "review_summary",
    "set_image_status",
    "update_annotation",
]
