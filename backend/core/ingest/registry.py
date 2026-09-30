"""适配器注册表：格式自动探测与统一入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..ir import DatasetBundle, make_uid, merge_bundles
from .base import BaseAdapter
from .coco import CocoAdapter
from .labelme import LabelMeAdapter
from .openlabel import OpenLabelAdapter
from .voc import VocAdapter
from .yolo import YoloAdapter

# 注册顺序 = 探测优先级
_ADAPTERS: List[Type[BaseAdapter]] = []


def register(adapter_cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
    if adapter_cls not in _ADAPTERS:
        _ADAPTERS.append(adapter_cls)
    return adapter_cls


def available_adapters() -> List[Dict[str, str]]:
    return [
        {"name": a.name, "display_name": a.display_name, "extensions": ",".join(a.extensions)}
        for a in _ADAPTERS
    ]


def get_adapter(name: str) -> Type[BaseAdapter]:
    for a in _ADAPTERS:
        if a.name == name:
            return a
    raise KeyError(f"未注册的适配器: {name}；可用: {[a.name for a in _ADAPTERS]}")


def detect_format(root) -> Optional[str]:
    """探测目录属于哪种格式，返回适配器 name；无法识别返回 None。"""
    for a in _ADAPTERS:
        try:
            if a.detect(root):
                return a.name
        except Exception:
            continue
    return None


def load_dataset(root, fmt: Optional[str] = None, **options) -> DatasetBundle:
    """统一入口：读入任意来源 -> DatasetBundle。

    参数
        root: 数据目录
        fmt:  强制指定格式（不传则自动探测）
        options: 透传给适配器，如 source_id / group_by
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    fmt = fmt or detect_format(root)
    if not fmt:
        raise ValueError(
            f"无法识别数据格式: {root}；请用 fmt 参数显式指定，"
            f"可选 {[a.name for a in _ADAPTERS]}"
        )
    return get_adapter(fmt).load(root, **options)


def load_and_merge(specs: List[Dict[str, Any]], merged_source_id: str = "merged") -> DatasetBundle:
    """加载多个来源并合并为一个数据集。

    这是「合并不同格式数据」的入口：每个 spec 可以指定不同的格式与解析参数。

        specs = [
            {"path": r"D:\\ds_roboflow"},                      # 自动探测 YOLO
            {"path": r"D:\\dmd", "level": "driver_actions"},   # OpenLABEL
        ]
        bundle = load_and_merge(specs)
    """
    if not specs:
        raise ValueError("至少需要一个数据来源")

    bundles: List[DatasetBundle] = []
    for raw in specs:
        spec = dict(raw)
        path = spec.pop("path", None)
        if not path:
            raise ValueError(f"数据来源缺少 path: {raw}")
        bundles.append(load_dataset(path, **spec))

    if len(bundles) == 1:
        return bundles[0]

    merged = merge_bundles(bundles)
    merged.source_id = merged_source_id
    merged.format_name = "merged"
    merged.root = " | ".join(b.root for b in bundles)
    merged.meta["sources"] = [
        {
            "source_id": b.source_id,
            "format": b.format_name,
            "root": b.root,
            "num_images": len(b.images),
            "num_annotations": len(b.annotations),
        }
        for b in bundles
    ]
    return merged


# 内置适配器注册。顺序即探测优先级：判定条件越明确的格式越靠前。
# OpenLABEL 要求 json 中必须含 "openlabel" 键；
# COCO 要求同时含 images/annotations/categories；
# VOC 要求存在 Annotations 目录且含 XML；
# YOLO 的判定最宽松（data.yaml 或 images+labels），排最后兜底。
register(OpenLabelAdapter)
register(CocoAdapter)
register(VocAdapter)
register(LabelMeAdapter)
register(YoloAdapter)
