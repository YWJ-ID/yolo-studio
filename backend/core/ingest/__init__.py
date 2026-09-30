"""接入层：把各种数据格式读成统一 IR。"""

from __future__ import annotations

from .base import BaseAdapter, names_from_yaml, normalize_category_name, read_yolo_label_file
from .coco import CocoAdapter
from .labelme import LabelMeAdapter
from .openlabel import OpenLabelAdapter, extract_frame_index
from .registry import (
    available_adapters,
    detect_format,
    get_adapter,
    load_and_merge,
    load_dataset,
    register,
)
from .voc import VocAdapter
from .yolo import YoloAdapter

__all__ = [
    "BaseAdapter",
    "CocoAdapter",
    "LabelMeAdapter",
    "OpenLabelAdapter",
    "VocAdapter",
    "YoloAdapter",
    "available_adapters",
    "detect_format",
    "extract_frame_index",
    "get_adapter",
    "load_and_merge",
    "load_dataset",
    "names_from_yaml",
    "normalize_category_name",
    "read_yolo_label_file",
    "register",
]
