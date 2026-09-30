"""YOLO Studio 核心库（不依赖 FastAPI，可独立 CLI 使用）。

分层约束：
    core  ← api / cli / web
    core 绝不反向依赖上层。
"""

from __future__ import annotations

from .ir import (
    KIND_BBOX,
    KIND_IMAGE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    VALID_SPLITS,
    Annotation,
    BBox,
    Category,
    DatasetBundle,
    ImageRecord,
    make_uid,
    merge_bundles,
    normalize_split,
)

__version__ = "0.1.0"

__all__ = [
    "KIND_BBOX",
    "KIND_IMAGE",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "VALID_SPLITS",
    "Annotation",
    "BBox",
    "Category",
    "DatasetBundle",
    "ImageRecord",
    "__version__",
    "make_uid",
    "merge_bundles",
    "normalize_split",
]
