"""适配器基类与通用工具。

每个数据格式实现一个 Adapter，职责只有一件：
    把**任意来源**的数据读成 `DatasetBundle`（见 core/ir.py）。

适配器不得做分类映射、不得做清洗、不得做划分——那些是后续模块的事。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ir import DatasetBundle, ImageRecord, make_uid


class BaseAdapter(ABC):
    """所有格式适配器的父类。"""

    name: str = "base"
    display_name: str = "未知格式"
    extensions: Tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def detect(cls, root) -> bool:
        """启发式判断该目录是否属于本格式。"""

    @classmethod
    @abstractmethod
    def load(cls, root, **options) -> DatasetBundle:
        """读取并返回 DatasetBundle。"""

    # ---------- 给子类复用的小工具 ----------

    @staticmethod
    def build_image(
        path,
        source_id: str,
        root,
        split: Optional[str] = None,
        group: Optional[str] = None,
        probe_size_fn=None,
    ) -> ImageRecord:
        """构造 ImageRecord，自动探测尺寸并生成稳定 uid。

        uid 里带上**根目录绝对路径**：多个来源合并时，不同目录下可能存在
        完全相同的相对路径（如都是 `train/images/000001.jpg`），
        只靠 rel_path 会撞车，导致合并时误判为重复。
        """
        p = Path(path).resolve()
        root_abs = Path(root).resolve()
        try:
            rel = str(p.relative_to(root_abs))
        except ValueError:
            rel = p.name

        width = height = 0
        if probe_size_fn is not None:
            size = probe_size_fn(p)
            if size:
                width, height = size

        uid = make_uid(source_id, str(root_abs), rel)
        return ImageRecord(
            uid=uid,
            path=str(p),
            rel_path=rel,
            width=width,
            height=height,
            source_id=source_id,
            split=split,
            group=group,
        )


def normalize_category_name(name: str) -> str:
    """把类别名做**轻度**归一：去首尾空白、合并内部连续空白。

    注意：这里不做大小写/下划线转换——那属于 taxonomy 映射模块，
    必须留下原始形态供人工确认，否则会静默合并掉不该合并的类。
    """
    return " ".join(str(name).strip().split())


def parse_data_yaml(path) -> Dict[str, Any]:
    """解析 data.yaml / dataset.yaml，返回原始 dict。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def names_from_yaml(data: Dict[str, Any]) -> Dict[int, str]:
    """从 data.yaml 的 names 字段提取 {id: 类名}。

    ultralytics 支持三种写法：
        names: [cat, dog]
        names: {0: cat, 1: dog}
        names: {'0': cat}  (yaml 字符串键)
    """
    names = data.get("names")
    result: Dict[int, str] = {}
    if names is None:
        return result
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                result[int(k)] = normalize_category_name(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            result[i] = normalize_category_name(v)
    return result


def read_yolo_label_file(txt_path) -> List[Tuple[int, float, float, float, float]]:
    """读取 YOLO 标签文件，返回 [(class_id, xc, yc, w, h)] 归一化坐标。

    容错处理：跳过空行与格式错误的行（问题由清洗模块统一汇总）。
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    try:
        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return rows

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        rows.append((cid, xc, yc, w, h))
    return rows
