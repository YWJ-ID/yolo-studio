"""统一中间表示（Canonical IR）。

设计约束（全项目最重要的约定）：
    1. 所有格式适配器必须先把数据归一到本文件的模型，**绝对像素坐标 xyxy**。
    2. 类别一律使用**字符串名**，不使用数字 id（不同数据集 id 含义不同）。
    3. 只在导出阶段（core/export）做一次 YOLO 归一化，中间过程绝不来回转换。

坐标约定：
    bbox 使用 xyxy 绝对像素坐标，原点在图像左上角，x 向右，y 向下。
    越界/负值由清洗模块负责发现，IR 层不做静默裁剪。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 标注形态
KIND_BBOX = "bbox"      # 目标检测：带边界框
KIND_IMAGE = "image"    # 图像分类：只标整图所属类别（如 DMD 的逐帧动作）

# 数据集划分的合法取值
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
VALID_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# 划分相关的常见别名（不同来源写法不同）
_SPLIT_ALIASES = {
    "train": SPLIT_TRAIN,
    "training": SPLIT_TRAIN,
    "valid": SPLIT_VAL,
    "val": SPLIT_VAL,
    "validation": SPLIT_VAL,
    "test": SPLIT_TEST,
    "testing": SPLIT_TEST,
}


def normalize_split(value: Optional[str]) -> Optional[str]:
    """把各种划分写法归一到 train/val/test；无法识别时返回 None。"""
    if not value:
        return None
    return _SPLIT_ALIASES.get(str(value).strip().lower())


def make_uid(*parts: str) -> str:
    """生成稳定的短 uid（用于图像/标注的全局唯一标识）。"""
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class BBox:
    """绝对像素坐标的边界框。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BBox":
        return cls(d["x1"], d["y1"], d["x2"], d["y2"])


@dataclass
class ImageRecord:
    """一张图像。"""

    uid: str
    path: str                 # 绝对路径
    width: int
    height: int
    source_id: str            # 来源标识（数据集/导入批次）
    rel_path: str = ""        # 相对来源根目录的路径
    split: Optional[str] = None
    group: Optional[str] = None    # 分组键：同一视频/序列的帧共用，用于防泄漏划分
    sha1: Optional[str] = None     # 精确重复检测
    phash: Optional[str] = None    # 近似重复检测
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ext(self) -> str:
        return Path(self.path).suffix.lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "path": self.path,
            "rel_path": self.rel_path,
            "width": self.width,
            "height": self.height,
            "source_id": self.source_id,
            "split": self.split,
            "group": self.group,
            "sha1": self.sha1,
            "phash": self.phash,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Annotation:
    """一条标注。

    两种形态（由 kind 区分）：
        KIND_BBOX  : 目标检测，bbox 必须有值
        KIND_IMAGE : 图像分类，bbox 为 None，只表示整图所属类别

    时序数据（如 OpenLABEL 的 frame_intervals）在接入阶段就展开为逐帧的图像级标注，
    因此 IR 本身不保存时间轴概念，只保留最终的“帧 → 类别”结果。
    """

    image_uid: str
    category: str             # 类别名（字符串，非 id）
    bbox: Optional[BBox] = None
    kind: str = KIND_BBOX
    segmentation: Optional[List[List[float]]] = None   # 多边形（LabelMe/COCO 提供时保留）
    score: Optional[float] = None                       # 预测框才有
    difficult: bool = False
    truncated: bool = False
    group: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_bbox(self) -> bool:
        return self.kind == KIND_BBOX and self.bbox is not None

    @property
    def area(self) -> float:
        return self.bbox.area if self.bbox else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_uid": self.image_uid,
            "category": self.category,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "kind": self.kind,
            "segmentation": self.segmentation,
            "score": self.score,
            "difficult": self.difficult,
            "truncated": self.truncated,
            "group": self.group,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Annotation":
        raw_bbox = d.get("bbox")
        return cls(
            image_uid=d["image_uid"],
            category=d["category"],
            bbox=BBox.from_dict(raw_bbox) if raw_bbox else None,
            kind=d.get("kind", KIND_BBOX),
            segmentation=d.get("segmentation"),
            score=d.get("score"),
            difficult=d.get("difficult", False),
            truncated=d.get("truncated", False),
            group=d.get("group"),
            meta=d.get("meta", {}),
        )


@dataclass
class Category:
    """一个类别在某个来源中的定义，用于后续映射到统一类别体系。"""

    name: str                          # 归一化后的名字
    original_name: str = ""            # 原始写法，便于追溯
    aliases: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_name": self.original_name,
            "aliases": self.aliases,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Category":
        return cls(
            name=d["name"],
            original_name=d.get("original_name", d["name"]),
            aliases=d.get("aliases", []),
            meta=d.get("meta", {}),
        )


@dataclass
class DatasetBundle:
    """一次导入产出的数据集包（内存态），是各模块之间的唯一交换格式。"""

    source_id: str
    format_name: str = "unknown"
    root: str = ""
    images: Dict[str, ImageRecord] = field(default_factory=dict)
    annotations: List[Annotation] = field(default_factory=list)
    categories: Dict[str, Category] = field(default_factory=dict)
    # 接入阶段发现的问题（清洗模块会做更深的检查）
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # uid -> 标注 的惰性索引。清洗/导出会频繁按图取标注，
    # 若每次线性扫描，万级数据集会退化成上亿次操作。
    _annotation_index: Optional[Dict[str, List[Annotation]]] = field(
        default=None, init=False, repr=False, compare=False
    )

    # ---------- 增删 ----------

    def add_image(self, image: ImageRecord) -> ImageRecord:
        self.images[image.uid] = image
        return image

    def add_annotation(self, ann: Annotation) -> Annotation:
        self.annotations.append(ann)
        self.categories.setdefault(ann.category, Category(name=ann.category))
        if self._annotation_index is not None:
            self._annotation_index.setdefault(ann.image_uid, []).append(ann)
        return ann

    def add_category(self, category: Category) -> None:
        self.categories.setdefault(category.name, category)

    def invalidate_index(self) -> None:
        self._annotation_index = None

    def annotation_index(self) -> Dict[str, List[Annotation]]:
        """按图像 uid 索引标注（惰性构建并缓存）。"""
        if self._annotation_index is None:
            index: Dict[str, List[Annotation]] = {}
            for a in self.annotations:
                index.setdefault(a.image_uid, []).append(a)
            self._annotation_index = index
        return self._annotation_index

    # ---------- 查询 ----------

    def annotations_of(self, image_uid: str) -> List[Annotation]:
        return self.annotation_index().get(image_uid, [])

    # ---------- 删除（供清洗模块使用） ----------

    def remove_image(self, uid: str) -> int:
        """删除图像及其全部标注，返回被一并删除的标注数。"""
        if uid not in self.images:
            return 0
        del self.images[uid]
        index = self.annotation_index()
        removed = len(index.pop(uid, []))
        if removed:
            self.annotations = [a for a in self.annotations if a.image_uid != uid]
        return removed

    def remove_annotation(self, target: Annotation) -> bool:
        """删除单条标注。"""
        try:
            self.annotations.remove(target)
        except ValueError:
            return False
        index = self.annotation_index()
        bucket = index.get(target.image_uid)
        if bucket:
            try:
                bucket.remove(target)
            except ValueError:
                pass
        return True

    def images_of_split(self, split: str) -> List[ImageRecord]:
        return [im for im in self.images.values() if im.split == split]

    def category_names(self) -> List[str]:
        """返回类别名列表。

        顺序优先采用 `categories` 的登记顺序——对 YOLO 适配器而言那正是 `data.yaml`
        里声明的下标顺序，保持它才能让导出后的类别下标与原数据集一致，
        否则用旧权重继续训练会发生类别错位。
        标注中出现但未登记的类别追加在末尾，保证不丢类。
        """
        seen: Dict[str, None] = {}
        for name in self.categories:
            seen.setdefault(name, None)
        for ann in self.annotations:
            seen.setdefault(ann.category, None)
        return list(seen.keys())

    def count_by_category(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ann in self.annotations:
            counts[ann.category] = counts.get(ann.category, 0) + 1
        return counts

    # ---------- 统计 ----------

    def stats(self) -> Dict[str, Any]:
        split_counts = {s: 0 for s in VALID_SPLITS}
        unassigned = 0
        for im in self.images.values():
            if im.split in split_counts:
                split_counts[im.split] += 1
            else:
                unassigned += 1

        num_bbox = sum(1 for a in self.annotations if a.is_bbox)
        num_image = len(self.annotations) - num_bbox
        groups = {im.group for im in self.images.values() if im.group}
        if num_bbox and num_image:
            annotation_kind = "mixed"
        elif num_image:
            annotation_kind = KIND_IMAGE
        elif num_bbox:
            annotation_kind = KIND_BBOX
        else:
            annotation_kind = "unknown"

        return {
            "format": self.format_name,
            "root": self.root,
            "annotation_kind": annotation_kind,
            "num_images": len(self.images),
            "num_annotations": len(self.annotations),
            "num_bbox_annotations": num_bbox,
            "num_image_labels": num_image,
            "num_categories": len(self.category_names()),
            "num_groups": len(groups),
            "split_counts": split_counts,
            "unassigned_images": unassigned,
            "count_by_category": self.count_by_category(),
            "warnings": len(self.warnings),
        }

    # ---------- 合并 ----------

    def merge(self, other: "DatasetBundle") -> "DatasetBundle":
        """把另一个来源合并进当前包。

        uid 冲突分两种情况：
          * 同一个文件被重复导入 -> 跳过并记录；
          * 不同文件恰好算出相同 uid -> 重新分配 uid，并同步改写其标注的引用。
        绝不能默默丢弃第二种——那会丢掉真实数据。
        """
        remap: Dict[str, Optional[str]] = {}

        for uid, im in other.images.items():
            if uid not in self.images:
                self.images[uid] = im
                continue

            if self.images[uid].path == im.path:
                self.warnings.append(f"同一文件被重复导入，已跳过: {im.path}")
                remap[uid] = None
                continue

            new_uid = make_uid("dedup", im.path)
            while new_uid in self.images:
                new_uid = make_uid("dedup", im.path, new_uid)
            im.uid = new_uid
            self.images[new_uid] = im
            remap[uid] = new_uid
            self.warnings.append(f"uid 冲突，已重新分配: {im.path} -> {new_uid}")

        for a in other.annotations:
            target = remap.get(a.image_uid, a.image_uid)
            if target is None:
                continue
            a.image_uid = target
            self.annotations.append(a)

        for name, cat in other.categories.items():
            self.categories.setdefault(name, cat)
        self.warnings.extend(other.warnings)
        self.invalidate_index()
        return self

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "format_name": self.format_name,
            "root": self.root,
            "images": [im.to_dict() for im in self.images.values()],
            "annotations": [a.to_dict() for a in self.annotations],
            "categories": [c.to_dict() for c in self.categories.values()],
            "warnings": self.warnings,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetBundle":
        bundle = cls(
            source_id=d.get("source_id", ""),
            format_name=d.get("format_name", "unknown"),
            root=d.get("root", ""),
            warnings=list(d.get("warnings", [])),
            meta=dict(d.get("meta", {})),
        )
        for im in d.get("images", []):
            rec = ImageRecord.from_dict(im)
            bundle.images[rec.uid] = rec
        bundle.annotations = [Annotation.from_dict(a) for a in d.get("annotations", [])]
        for c in d.get("categories", []):
            cat = Category.from_dict(c)
            bundle.categories[cat.name] = cat
        return bundle

    def save_json(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path) -> "DatasetBundle":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def merge_bundles(bundles: Iterable[DatasetBundle]) -> DatasetBundle:
    """合并多个来源的数据包。"""
    result: Optional[DatasetBundle] = None
    for b in bundles:
        if result is None:
            result = b
        else:
            result.merge(b)
    return result or DatasetBundle(source_id="empty")
