"""M7 预标注：模型批量打框 → 统一 IR → 复用清洗/类别/划分/导出。

产出的是**伪标签**，必须人工复核后才可作为训练数据。
详见本模块 `prelabel.py` 顶部的定位说明。
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
