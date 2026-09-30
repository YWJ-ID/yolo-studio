"""Pascal VOC 格式适配器。

目录结构：

    <root>/
    ├── Annotations/*.xml          # 每个 XML 一张图
    ├── JPEGImages/*.jpg
    └── ImageSets/Main/
        ├── train.txt              # 每行一个图像 id（不含扩展名）
        ├── val.txt
        └── test.txt

XML 内容：

    <annotation>
      <filename>000001.jpg</filename>
      <size><width>640</width><height>480</height><depth>3</depth></size>
      <object>
        <name>cat</name>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox>
      </object>
    </annotation>

## 关于坐标基准（重要）

Pascal VOC 原始规范里坐标是 **1-based 且 xmax/ymax 为闭区间**；
但实际数据里大量使用 LabelImg 产出，而 LabelImg 写的是 **0-based** 坐标。
两者混用会导致整体偏移 1 像素。

因此本适配器默认**直接采用 XML 中的数值**（与主流转换工具一致），
若确认数据来自原始 VOC devkit，可传 `voc_one_based=True` 做 `-1` 修正。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from ..io import IMAGE_EXTS, probe_size
from ..ir import Annotation, BBox, Category, DatasetBundle, ImageRecord, make_uid, normalize_split
from .base import BaseAdapter, normalize_category_name

_ANNOTATION_DIRS = ("Annotations", "annotations", "Annotation")
_IMAGE_DIRS = ("JPEGImages", "images", "Images")
_IMAGESET_DIRS = ("ImageSets/Main", "ImageSets", "imagesets/Main")


class VocAdapter(BaseAdapter):
    name = "voc"
    display_name = "Pascal VOC XML"
    extensions = (".xml",)

    # ---------- 探测 ----------

    @classmethod
    def detect(cls, root) -> bool:
        root = Path(root)
        if not root.is_dir():
            return False
        annotations = cls._find_dir(root, _ANNOTATION_DIRS)
        if annotations is None:
            return False
        return next(annotations.glob("*.xml"), None) is not None

    # ---------- 读取 ----------

    @classmethod
    def load(
        cls,
        root,
        source_id: Optional[str] = None,
        voc_one_based: bool = False,
        **options,
    ) -> DatasetBundle:
        root = Path(root).resolve()
        source_id = source_id or f"voc:{root.name}"

        bundle = DatasetBundle(source_id=source_id, format_name=cls.name, root=str(root))
        bundle.meta["voc_one_based"] = bool(voc_one_based)

        annotations_dir = cls._find_dir(root, _ANNOTATION_DIRS)
        if annotations_dir is None:
            bundle.warnings.append(f"未找到 Annotations 目录: {root}")
            return bundle

        images_dir = cls._find_dir(root, _IMAGE_DIRS) or root
        split_map = cls._read_image_sets(root)

        xml_files = sorted(annotations_dir.glob("*.xml"))
        if not xml_files:
            bundle.warnings.append(f"Annotations 目录下没有 XML: {annotations_dir}")
            return bundle

        for xml_path in xml_files:
            cls._load_one(bundle, root, xml_path, images_dir, source_id, split_map, voc_one_based)

        bundle.meta["image_sets"] = {k: len(v) for k, v in split_map.items()}
        return bundle

    # ---------- 内部 ----------

    @classmethod
    def _load_one(cls, bundle, root, xml_path, images_dir, source_id, split_map, one_based):
        try:
            tree = ET.parse(xml_path)
            node = tree.getroot()
        except Exception as exc:
            bundle.warnings.append(f"{xml_path.name} 解析失败: {exc}")
            return

        filename = (node.findtext("filename") or "").strip()
        if not filename:
            filename = xml_path.stem + ".jpg"

        image_path = cls._resolve_image(images_dir, root, xml_path, filename)
        if image_path is None:
            bundle.warnings.append(f"找不到图像文件: {filename}（来自 {xml_path.name}）")
            return

        size = node.find("size")
        width = _to_int(size.findtext("width")) if size is not None else 0
        height = _to_int(size.findtext("height")) if size is not None else 0
        if width <= 0 or height <= 0:
            probed = probe_size(image_path)
            if probed:
                width, height = probed
            else:
                bundle.warnings.append(f"无法确定图像尺寸: {filename}")

        # 划分优先级：ImageSets > XML 的 folder 字段
        split = split_map.get(xml_path.stem)
        if split is None:
            folder = (node.findtext("folder") or "").strip()
            split = normalize_split(folder) if folder else None

        try:
            rel = str(image_path.relative_to(root))
        except ValueError:
            rel = image_path.name

        uid = make_uid(source_id, str(root), rel)
        image = ImageRecord(
            uid=uid,
            path=str(image_path),
            rel_path=rel,
            width=width,
            height=height,
            source_id=source_id,
            split=split,
            group=_group_from_rel(rel),
        )
        bundle.add_image(image)

        for obj in node.findall("object"):
            name = normalize_category_name(obj.findtext("name") or "")
            if not name:
                continue
            bundle.add_category(Category(name=name, original_name=name))

            box = obj.find("bndbox")
            if box is None:
                bundle.warnings.append(f"{xml_path.name}: object 缺少 bndbox")
                continue

            xmin = _to_float(box.findtext("xmin"))
            ymin = _to_float(box.findtext("ymin"))
            xmax = _to_float(box.findtext("xmax"))
            ymax = _to_float(box.findtext("ymax"))
            if None in (xmin, ymin, xmax, ymax):
                bundle.warnings.append(f"{xml_path.name}: bndbox 坐标不完整")
                continue

            if one_based:
                xmin, ymin = xmin - 1.0, ymin - 1.0

            bundle.add_annotation(
                Annotation(
                    image_uid=uid,
                    category=name,
                    bbox=BBox(xmin, ymin, xmax, ymax),
                    difficult=(obj.findtext("difficult") or "0").strip() == "1",
                    truncated=(obj.findtext("truncated") or "0").strip() == "1",
                    meta={"source_xml": xml_path.name},
                )
            )

    # ---------- 工具 ----------

    @staticmethod
    def _find_dir(root: Path, names) -> Optional[Path]:
        for name in names:
            candidate = root / name
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _read_image_sets(root: Path) -> Dict[str, str]:
        """读取 ImageSets/Main/*.txt，返回 {图像id: 划分}。"""
        mapping: Dict[str, str] = {}
        for sub in _IMAGESET_DIRS:
            base = root / sub
            if not base.is_dir():
                continue
            for txt in sorted(base.glob("*.txt")):
                split = normalize_split(txt.stem)
                if not split:
                    continue
                try:
                    text = txt.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for line in text.splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    # 常见格式：`000001` 或 `000001 1`（后者是难例标记）
                    mapping.setdefault(parts[0], split)
            if mapping:
                break
        return mapping

    @staticmethod
    def _resolve_image(images_dir: Path, root: Path, xml_path: Path, filename: str):
        rel = Path(filename)
        candidates = [
            images_dir / rel,
            images_dir / rel.name,
            root / rel,
            xml_path.parent / rel,
            xml_path.parent / rel.name,
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        bases = [images_dir, root, xml_path.parent]
        for base in bases:
            if not base.is_dir():
                continue
            for ext in IMAGE_EXTS:
                c = base / (rel.stem + ext)
                if c.is_file():
                    return c.resolve()
        return None


# ---------------------------------------------------------------------------


def _to_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_from_rel(rel: str) -> Optional[str]:
    """用相对路径中的目录层级作为分组键（同目录多为同一场景/视频）。"""
    parts = Path(rel).parts
    return str(Path(*parts[:-1])) if len(parts) > 1 else None
