"""COCO JSON 格式适配器。

支持两种常见布局：

    A) 一个 JSON 存一个划分
       <root>/
       ├── annotations/instances_train2017.json
       ├── annotations/instances_val2017.json
       ├── train2017/*.jpg
       └── val2017/*.jpg

    B) 单个 JSON
       <root>/
       ├── annotations.json
       └── images/*.jpg

COCO 的 bbox 是 `[x, y, width, height]`，**左上角原点、绝对像素**，
与 IR 的绝对像素 xyxy 只差一次换算，不涉及归一化，因此不会丢失精度。

划分来源优先级：
    1. 显式参数 split
    2. 标注文件名（instances_train2017.json -> train）
    3. 图像 file_name 的路径前缀（train2017/xxx.jpg -> train）
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..io import is_image, probe_size
from ..ir import (
    KIND_BBOX,
    Annotation,
    BBox,
    Category,
    DatasetBundle,
    ImageRecord,
    make_uid,
    normalize_split,
)
from .base import BaseAdapter, normalize_category_name

# 从文件名/路径中识别划分的关键词
_SPLIT_RE = re.compile(r"(train|val|valid|validation|test)", re.IGNORECASE)

# 探测/加载时扫描 JSON 的最大深度（避免把逐图一 JSON 的 LabelMe 数据集扫穿）
_MAX_JSON_DEPTH = 2


class CocoAdapter(BaseAdapter):
    name = "coco"
    display_name = "COCO JSON"
    extensions = (".json",)

    # ---------- 探测 ----------

    @classmethod
    def detect(cls, root) -> bool:
        return bool(cls.find_annotation_files(Path(root)))

    @classmethod
    def find_annotation_files(cls, root: Path) -> List[Path]:
        """找出 COCO 标注文件：同时含 images / annotations / categories 的 JSON。"""
        if not root.is_dir():
            return []

        candidates: List[Path] = []
        candidates.extend(root.glob("*.json"))
        for i in range(1, _MAX_JSON_DEPTH):
            candidates.extend(root.glob("/".join(["*"] * i) + "/*.json"))

        found: List[Path] = []
        for p in sorted(set(candidates)):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if all(k in data for k in ("images", "annotations", "categories")):
                found.append(p)
        return found

    # ---------- 读取 ----------

    @classmethod
    def load(
        cls,
        root,
        source_id: Optional[str] = None,
        images_dir: Optional[str] = None,
        split: Optional[str] = None,
        **options,
    ) -> DatasetBundle:
        root = Path(root).resolve()
        source_id = source_id or f"coco:{root.name}"

        bundle = DatasetBundle(source_id=source_id, format_name=cls.name, root=str(root))
        ann_files = cls.find_annotation_files(root)
        if not ann_files:
            bundle.warnings.append(f"未找到 COCO 标注文件: {root}")
            return bundle

        explicit_split = normalize_split(split) if split else None
        for ann_path in ann_files:
            cls._load_one(bundle, root, ann_path, source_id, images_dir, explicit_split)

        return bundle

    # ---------- 内部 ----------

    @classmethod
    def _load_one(
        cls,
        bundle: DatasetBundle,
        root: Path,
        ann_path: Path,
        source_id: str,
        images_dir: Optional[str],
        explicit_split: Optional[str],
    ) -> None:
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as exc:
            bundle.warnings.append(f"{ann_path.name} 解析失败: {exc}")
            return

        # 1) 类别表（保持 JSON 中的顺序，让导出后的下标与原始一致）
        id_to_name: Dict[Any, str] = {}
        for cat in data.get("categories") or []:
            if not isinstance(cat, dict):
                continue
            name = normalize_category_name(cat.get("name", ""))
            if not name:
                continue
            id_to_name[cat.get("id")] = name
            bundle.add_category(
                Category(name=name, original_name=str(cat.get("name", "")),
                         meta={"coco_id": cat.get("id")})
            )

        file_split = explicit_split or cls._split_from_text(ann_path.stem)
        image_base = (root / images_dir) if images_dir else None

        # 2) 图像
        id_to_uid: Dict[Any, str] = {}
        for img in data.get("images") or []:
            if not isinstance(img, dict):
                continue
            file_name = str(img.get("file_name") or "")
            if not file_name:
                continue

            path = cls._resolve_image(root, ann_path, image_base, file_name)
            if path is None:
                bundle.warnings.append(f"找不到图像文件: {file_name}（来自 {ann_path.name}）")
                continue

            this_split = file_split or cls._split_from_text(file_name)
            width = int(img.get("width") or 0)
            height = int(img.get("height") or 0)
            if width <= 0 or height <= 0:
                probed = probe_size(path)
                if probed:
                    width, height = probed

            rel = file_name
            uid = make_uid(source_id, str(root), rel)
            image = ImageRecord(
                uid=uid,
                path=str(path),
                rel_path=rel,
                width=width,
                height=height,
                source_id=source_id,
                split=this_split,
                group=_group_from_path(file_name),
            )
            bundle.add_image(image)
            id_to_uid[img.get("id")] = uid

        # 3) 标注
        for ann in data.get("annotations") or []:
            if not isinstance(ann, dict):
                continue
            uid = id_to_uid.get(ann.get("image_id"))
            if uid is None:
                continue

            bbox = ann.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                bundle.warnings.append(f"标注缺少合法 bbox: {ann.get('id')}")
                continue

            name = id_to_name.get(ann.get("category_id"))
            if name is None:
                bundle.warnings.append(f"标注的 category_id 未在 categories 中登记: {ann.get('category_id')}")
                continue

            x, y, w, h = (float(v) for v in bbox[:4])
            bundle.add_annotation(
                Annotation(
                    image_uid=uid,
                    category=name,
                    bbox=BBox.from_xywh(x, y, w, h),
                    kind=KIND_BBOX,
                    segmentation=ann.get("segmentation") if isinstance(ann.get("segmentation"), list) else None,
                    group="iscrowd" if ann.get("iscrowd") else None,
                    meta={"coco_id": ann.get("id"), "iscrowd": ann.get("iscrowd", 0)},
                )
            )

        bundle.meta.setdefault("annotation_files", []).append(
            {
                "file": ann_path.name,
                "split": file_split,
                "num_images": len(data.get("images") or []),
                "num_annotations": len(data.get("annotations") or []),
            }
        )

    # ---------- 工具 ----------

    @staticmethod
    def _split_from_text(text: str) -> Optional[str]:
        match = _SPLIT_RE.search(text or "")
        return normalize_split(match.group(1)) if match else None

    @staticmethod
    def _resolve_image(
        root: Path,
        ann_path: Path,
        image_base: Optional[Path],
        file_name: str,
    ) -> Optional[Path]:
        """在若干候选位置里找图像。"""
        rel = Path(file_name)

        candidates: List[Path] = []
        if image_base is not None:
            candidates.append(image_base / rel)
        candidates.append(ann_path.parent / rel)          # 与标注同目录
        candidates.append(root / rel)                      # 相对数据集根
        candidates.append(root / "images" / rel)           # 常见的 images/ 容器
        # file_name 带子目录时，再尝试只用文件名
        candidates.append(ann_path.parent / rel.name)
        candidates.append(root / "images" / rel.name)

        for c in candidates:
            if c.is_file():
                return c.resolve()
        return None


def _group_from_path(file_name: str) -> Optional[str]:
    """用 file_name 中的目录层级作为分组键。

    COCO 里同一段视频/场景的图常放在同一子目录，
    这样划分时同目录的图会整体落到同一子集，避免相邻帧泄漏。
    """
    parts = Path(file_name).parts
    if len(parts) <= 1:
        return None
    return str(Path(*parts[:-1]))
