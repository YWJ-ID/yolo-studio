"""M7 预标注：模型批量打框 → 统一 IR → 复用清洗/类别/划分/导出。

产出的是**伪标签**，必须人工复核后才可作为训练数据。
详见本模块 `prelabel.py` 顶部的定位说明。

复核 / 编辑（人工修正伪标签）见 `review.py`。
"""

from __future__ import annotations

from .prelabel import (
    PRELABEL_PREFIX,
    SOURCE_MODEL,
    PrelabelConfig,
    PrelabelReport,
    annotate_bundle,
    attach_lineage,
    build_bundle,
    ensure_prelabel_prefix,
    export_name,
    list_images,
    run_prelabel,
)
from .review import (
    REVIEW_CONFIRMED,
    REVIEW_PENDING,
    SOURCE_HUMAN,
    ReviewError,
    add_annotation,
    attach_review_summary,
    find_annotation,
    image_review_status,
    next_annotation_id,
    remove_annotation,
    review_summary,
    set_image_status,
    update_annotation,
)

__all__ = [
    "PRELABEL_PREFIX",
    "REVIEW_CONFIRMED",
    "REVIEW_PENDING",
    "SOURCE_HUMAN",
    "SOURCE_MODEL",
    "PrelabelConfig",
    "PrelabelReport",
    "ReviewError",
    "add_annotation",
    "annotate_bundle",
    "attach_lineage",
    "attach_review_summary",
    "build_bundle",
    "ensure_prelabel_prefix",
    "export_name",
    "find_annotation",
    "image_review_status",
    "list_images",
    "next_annotation_id",
    "remove_annotation",
    "review_summary",
    "run_prelabel",
    "set_image_status",
    "update_annotation",
]
