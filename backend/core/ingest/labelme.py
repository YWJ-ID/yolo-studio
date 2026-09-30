"""LabelMe JSON 格式适配器。

每张图对应一个同名 JSON：

    <root>/
    ├── img_001.jpg
    ├── img_001.json
    └── ...

JSON 内容：

    {
      "version": "5.1.1",
      "imagePath": "img_001.jpg",
      "imageWidth": 640, "imageHeight": 480,
      "shapes": [
        {"label": "cat", "shape_type": "rectangle",
         "points": [[x1,y1],[x2,y2]], "flags": {}},
        {"label": "dog", "shape_type": "polygon",
         "points": [[x,y], ...]}
      ]
    }

支持 `rectangle` / `polygon` / `circle` 三种形状：
    * rectangle -> 直接取两角点为 xyxy（自动纠正顺序）
    * polygon   -> 取点集的外接矩形，同时保留多边形作为 segmentation
    * circle    -> 由圆心与半径换算外接矩形

`line` / `point` 等无法构成检测框的形状会被跳过并记录在警告里。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..io import IMAGE_EXTS, probe_size
from ..ir import Annotation, BBox, Category, DatasetBundle, ImageRecord, make_uid, normalize_split
from .base import BaseAdapter, normalize_category_name

_SUPPORTED_SHAPES = ("rectangle", "polygon", "circle")


class LabelMeAdapter(BaseAdapter):
    name = "labelme"
    display_name = "LabelMe JSON"
    extensions = (".json",)

    # ---------- 探测 ----------

    @classmethod
    def detect(cls, root) -> bool:
        return bool(cls.find_label_files(Path(root)))

    @classmethod
    def find_label_files(cls, root: Path) -> List[Path]:
        """找出 LabelMe 标注：含 shapes 且带 imagePath/imageWidth 的 JSON。"""
        if not root.is_dir():
            return []

        candidates: List[Path] = []
        candidates.extend(root.glob("*.json"))
        candidates.extend(root.glob("*/*.json"))

        found: List[Path] = []
        for p in sorted(set(candidates)):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if "shapes" in data and ("imagePath" in data or "imageWidth" in data):
                found.append(p)
        return found

    # ---------- 读取 ----------

    @classmethod
    def load(
        cls,
        root,
        source_id: Optional[str] = None,
        **options,
    ) -> DatasetBundle:
        root = Path(root).resolve()
        source_id = source_id or f"labelme:{root.name}"

        bundle = DatasetBundle(source_id=source_id, format_name=cls.name, root=str(root))
        label_files = cls.find_label_files(root)
        if not label_files:
            bundle.warnings.append(f"未找到 LabelMe 标注文件: {root}")
            return bundle

        skipped_shapes: Dict[str, int] = {}
        for json_path in label_files:
            cls._load_one(bundle, root, json_path, source_id, skipped_shapes)

        if skipped_shapes:
            bundle.warnings.append(
                f"跳过了无法构成检测框的形状: {skipped_shapes}"
            )
        return bundle

    # ---------- 内部 ----------

    @classmethod
    def _load_one(cls, bundle, root, json_path, source_id, skipped_shapes) -> None:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            bundle.warnings.append(f"{json_path.name} 解析失败: {exc}")
            return

        image_path = cls._resolve_image(root, json_path, data)
        if image_path is None:
            bundle.warnings.append(f"找不到图像文件: {data.get('imagePath') or json_path.stem}")
            return

        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            probed = probe_size(image_path)
            if probed:
                width, height = probed
            else:
                bundle.warnings.append(f"无法确定图像尺寸: {json_path.name}")

        try:
            rel = str(image_path.relative_to(root))
        except ValueError:
            rel = image_path.name

        uid = make_uid(source_id, str(root), rel + "|" + json_path.name)
        image = ImageRecord(
            uid=uid,
            path=str(image_path),
            rel_path=rel,
            width=width,
            height=height,
            source_id=source_id,
            split=cls._split_from(dirname=rel),
            group=_group_from_rel(rel),
            meta={"labelme_json": json_path.name},
        )
        bundle.add_image(image)

        for shape in data.get("shapes") or []:
            if not isinstance(shape, dict):
                continue

            name = normalize_category_name(shape.get("label") or "")
            if not name:
                continue

            shape_type = str(shape.get("shape_type") or "polygon").lower()
            points = shape.get("points") or []
            if not isinstance(points, list) or len(points) < 2:
                skipped_shapes[shape_type] = skipped_shapes.get(shape_type, 0) + 1
                continue

            parsed = cls._bbox_from_shape(shape_type, points)
            if parsed is None:
                skipped_shapes[shape_type] = skipped_shapes.get(shape_type, 0) + 1
                continue

            bbox, segmentation = parsed
            bundle.add_category(Category(name=name, original_name=str(shape.get("label"))))
            bundle.add_annotation(
                Annotation(
                    image_uid=uid,
                    category=name,
                    bbox=bbox,
                    segmentation=segmentation,
                    meta={"shape_type": shape_type, "flags": shape.get("flags") or {}},
                )
            )

    # ---------- 形状换算 ----------

    @staticmethod
    def _bbox_from_shape(
        shape_type: str, points: List[Any]
    ) -> Optional[Tuple[BBox, Optional[List[List[float]]]]]:
        try:
            pts = [(float(p[0]), float(p[1])) for p in points]
        except (TypeError, ValueError, IndexError):
            return None

        if shape_type == "rectangle":
            (x1, y1), (x2, y2) = pts[0], pts[1]
            # LabelMe 不保证角点顺序，这里统一成左上/右下
            return BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), None

        if shape_type == "polygon":
            if len(pts) < 3:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            flat = [coord for p in pts for coord in p]
            return BBox(min(xs), min(ys), max(xs), max(ys)), [flat]

        if shape_type == "circle":
            (cx, cy), (ex, ey) = pts[0], pts[1]
            r = math.hypot(ex - cx, ey - cy)
            return BBox(cx - r, cy - r, cx + r, cy + r), None

        return None

    # ---------- 工具 ----------

    @staticmethod
    def _resolve_image(root: Path, json_path: Path, data: Dict[str, Any]) -> Optional[Path]:
        declared = str(data.get("imagePath") or "").strip()
        candidates: List[Path] = []

        if declared:
            rel = Path(declared.replace("\\", "/"))
            candidates.extend([
                json_path.parent / rel,
                json_path.parent / rel.name,
                root / rel,
                root / rel.name,
            ])

        stem = json_path.stem
        for base in (json_path.parent, root):
            for ext in IMAGE_EXTS:
                candidates.append(base / (stem + ext))
            # 标注放在 annotations/ 而图在 images/ 的布局
            candidates.append(base.parent / "images" / (stem + ".jpg"))

        for c in candidates:
            if c.is_file():
                return c.resolve()
        return None

    @staticmethod
    def _split_from(dirname: str) -> Optional[str]:
        """LabelMe 本身不记录划分，从目录名里碰运气。"""
        import re

        match = re.search(r"(train|val|valid|validation|test)", dirname or "", re.IGNORECASE)
        return normalize_split(match.group(1)) if match else None


def _group_from_rel(rel: str) -> Optional[str]:
    """用目录层级作为分组键（LabelMe 常用于视频抽帧，同目录多为一组）。"""
    parts = Path(rel).parts
    return str(Path(*parts[:-1])) if len(parts) > 1 else None
