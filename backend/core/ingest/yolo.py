"""YOLO 格式适配器（含 Roboflow / ultralytics 常见目录结构）。

支持的目录形态（自动探测，按优先级）：

    1) 显式 yaml 路径（最优先）
       data.yaml 里写 train: ../train/images / val: ... / test: ...

    2) 划分目录并列
       <root>/{train,valid,test}/images/*.jpg
       <root>/{train,valid,test}/labels/*.txt

    3) 反向划分（ultralytics 风格）
       <root>/images/{train,val,test}/
       <root>/labels/{train,val,test}/

    4) 扁平池（未划分，split=None，交由划分模块处理）
       <root>/images/*.jpg  +  <root>/labels/*.txt
       或 <root>/*.jpg + 同名 *.txt
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ir import Annotation, BBox, Category, DatasetBundle, ImageRecord, make_uid, normalize_split
from ..io import IMAGE_EXTS, is_image, probe_size
from .base import (
    BaseAdapter,
    names_from_yaml,
    normalize_category_name,
    parse_data_yaml,
    read_yolo_label_file,
)

_SPLIT_DIR_NAMES = ("train", "valid", "val", "test")


class YoloAdapter(BaseAdapter):
    name = "yolo"
    display_name = "YOLO (txt + data.yaml)"
    extensions = (".txt", ".yaml", ".yml")

    # ---------- 探测 ----------

    @classmethod
    def detect(cls, root) -> bool:
        root = Path(root)
        if not root.is_dir():
            return False
        # 有 data.yaml / dataset.yaml
        if _find_yaml(root) is not None:
            return True
        # 有 images/labels 结构
        if (root / "images").is_dir() and (root / "labels").is_dir():
            return True
        # 有 train/images
        for s in _SPLIT_DIR_NAMES:
            if (root / s / "images").is_dir():
                return True
            if (root / "images" / s).is_dir():
                return True
        return False

    # ---------- 读取 ----------

    @classmethod
    def load(cls, root, source_id: Optional[str] = None, group_by: Optional[str] = None,
             **options) -> DatasetBundle:
        root = Path(root).resolve()
        source_id = source_id or f"yolo:{root.name}"

        bundle = DatasetBundle(
            source_id=source_id,
            format_name=cls.name,
            root=str(root),
        )

        yaml_path = _find_yaml(root)
        yaml_data: Dict[str, Any] = {}
        names: Dict[int, str] = {}
        if yaml_path:
            try:
                yaml_data = parse_data_yaml(yaml_path)
                names = names_from_yaml(yaml_data)
                bundle.meta["yaml_path"] = str(yaml_path)
            except Exception as exc:
                bundle.warnings.append(f"data.yaml 解析失败: {exc}")

        # 登记类别（即使某类没有标注，也要保留，避免类别顺序漂移）
        for cid in sorted(names):
            cname = names[cid]
            bundle.add_category(Category(name=cname, original_name=cname, meta={"source_id": cid}))

        pairs = _discover_split_pairs(root, yaml_path, yaml_data)
        if not pairs:
            bundle.warnings.append(f"未发现任何图像/标签目录对: {root}")

        for split, images_dir, labels_dir in pairs:
            cls._load_split(bundle, source_id, root, split, images_dir, labels_dir, names, group_by)

        bundle.meta["discovered"] = [
            {"split": s, "images": str(i), "labels": str(l) if l else None} for s, i, l in pairs
        ]
        cls._scan_orphan_labels(bundle, pairs)
        return bundle

    @staticmethod
    def _scan_orphan_labels(bundle, pairs) -> None:
        """反向检查：存在标签文件但找不到对应图像。

        正向加载时这类问题不可见（IR 是从图像出发构建的），
        但它们是真实的数据缺陷，会表现为"某类样本莫名偏少"。
        """
        orphans: List[str] = []
        for _split, images_dir, labels_dir in pairs:
            if not labels_dir:
                continue
            labels_dir = Path(labels_dir)
            if not labels_dir.is_dir():
                continue
            for txt in labels_dir.rglob("*.txt"):
                try:
                    rel = txt.relative_to(labels_dir)
                except ValueError:
                    rel = Path(txt.name)
                candidates = [
                    (Path(images_dir) / rel).with_suffix(ext)
                    for ext in IMAGE_EXTS
                ]
                candidates.append(images_dir / rel.with_suffix(".jpg"))
                if not any(c.is_file() for c in candidates):
                    orphans.append(str(txt))
        if orphans:
            bundle.meta["orphan_labels"] = orphans

    # ---------- 内部 ----------

    @classmethod
    def _load_split(cls, bundle, source_id, root, split, images_dir, labels_dir, names, group_by):
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            bundle.warnings.append(f"图像目录不存在: {images_dir}")
            return

        for img_path in sorted(images_dir.rglob("*")):
            if not img_path.is_file() or not is_image(img_path):
                continue

            group = _derive_group(img_path, images_dir, group_by)
            image = cls.build_image(
                path=img_path,
                source_id=source_id,
                root=root,
                split=split,
                group=group,
                probe_size_fn=probe_size,
            )
            if image.width <= 0 or image.height <= 0:
                bundle.warnings.append(f"无法读取图像尺寸: {img_path}")
            bundle.add_image(image)

            label_path = _resolve_label_path(img_path, images_dir, labels_dir)
            if label_path is None or not label_path.exists():
                # 无标签：可能是背景图，也可能是漏标；交给清洗模块判定
                bundle.warnings.append(f"缺少标签文件: {img_path.name}")
                continue

            for cid, xc, yc, w, h in read_yolo_label_file(label_path):
                cname = names.get(cid)
                if cname is None:
                    bundle.warnings.append(
                        f"标签类别 id 越界 (id={cid}) 于 {label_path.name}，已跳过"
                    )
                    continue
                bbox = _yolo_norm_to_xyxy(xc, yc, w, h, image.width, image.height)
                bundle.add_annotation(
                    Annotation(
                        image_uid=image.uid,
                        category=cname,
                        bbox=bbox,
                        meta={"source_label": str(label_path)},
                    )
                )


# --------------------------------------------------------------------------
# 探测与路径推断
# --------------------------------------------------------------------------


def _find_yaml(root: Path) -> Optional[Path]:
    for candidate in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = root / candidate
        if p.is_file():
            return p
    # 兼容 Roboflow 偶发的嵌套/改名情况
    for p in sorted(root.glob("*.y*ml")):
        try:
            if "names" in (parse_data_yaml(p) or {}):
                return p
        except Exception:
            continue
    return None


def _discover_split_pairs(
    root: Path,
    yaml_path: Optional[Path],
    yaml_data: Dict[str, Any],
) -> List[Tuple[Optional[str], Path, Optional[Path]]]:
    """返回 [(split, images_dir, labels_dir)]。"""
    pairs: List[Tuple[Optional[str], Path, Optional[Path]]] = []

    # 策略 1：yaml 显式路径
    if yaml_data:
        base = yaml_path.parent if yaml_path else root
        found_explicit = False
        for key in ("train", "val", "valid", "test"):
            value = yaml_data.get(key)
            if not value or not isinstance(value, str):
                continue
            for raw in value.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                images_dir = (base / raw).resolve()
                if images_dir.is_dir():
                    pairs.append(
                        (normalize_split(key), images_dir, _infer_labels_dir(images_dir))
                    )
                    found_explicit = True
        if found_explicit:
            return _dedupe_pairs(pairs)

    # 策略 2：root/{split}/images
    for s in _SPLIT_DIR_NAMES:
        images_dir = root / s / "images"
        if images_dir.is_dir():
            pairs.append((normalize_split(s), images_dir, _infer_labels_dir(images_dir)))

    # 策略 3：root/images/{split}
    for s in _SPLIT_DIR_NAMES:
        images_dir = root / "images" / s
        if images_dir.is_dir():
            pairs.append((normalize_split(s), images_dir, root / "labels" / s))

    if pairs:
        return _dedupe_pairs(pairs)

    # 策略 4：扁平池
    flat_images = root / "images"
    if flat_images.is_dir():
        pairs.append((None, flat_images, _infer_labels_dir(flat_images)))
    elif any(is_image(p) for p in root.glob("*")):
        pairs.append((None, root, root))

    return _dedupe_pairs(pairs)


def _dedupe_pairs(pairs):
    seen = set()
    out = []
    for split, images_dir, labels_dir in pairs:
        key = (split, str(images_dir))
        if key in seen:
            continue
        seen.add(key)
        out.append((split, images_dir, labels_dir))
    return out


def _infer_labels_dir(images_dir: Path) -> Optional[Path]:
    """由图像目录推断标签目录。"""
    images_dir = Path(images_dir)

    # a) 同级 labels/
    sibling = images_dir.parent / "labels"
    if sibling.is_dir():
        return sibling

    # b) 路径中 images -> labels
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            candidate = Path(*parts)
            if candidate.is_dir():
                return candidate
            break

    # c) 标签与图像同目录（扁平 txt 场景）
    return images_dir


def _resolve_label_path(img_path: Path, images_dir: Path, labels_dir: Optional[Path]) -> Optional[Path]:
    """定位某张图对应的标签文件。"""
    try:
        rel = img_path.relative_to(images_dir)
    except ValueError:
        rel = Path(img_path.name)

    candidates = []
    if labels_dir is not None:
        candidates.append(Path(labels_dir) / rel.with_suffix(".txt"))
    candidates.append(img_path.with_suffix(".txt"))
    candidates.append(images_dir.parent / "labels" / rel.with_suffix(".txt"))

    for c in candidates:
        if c.is_file():
            return c
    return candidates[0] if candidates else None


def _derive_group(img_path: Path, images_dir: Path, group_by: Optional[str]) -> Optional[str]:
    """推导分组键（用于防止同视频帧跨集合泄漏）。

    group_by 支持：
        None / "none"        不分组
        "parent"             按图像所在子目录分组（视频抽帧常见）
        "stem"               按文件名去掉**末尾**数字分组（frame_0001 -> frame_）
        "regex:<pattern>"    用正则从文件名提取，取第一个捕获组；
                             例如 "-170-.*?-(\\d+)_mp4" 这类带帧号的命名
    """
    if not group_by or group_by == "none":
        return None

    if group_by.startswith("regex:"):
        pattern = group_by[len("regex:"):]
        try:
            match = re.search(pattern, img_path.stem)
        except re.error:
            return None
        if not match:
            return None
        return match.group(1) if match.groups() else match.group(0)

    if group_by == "parent":
        try:
            rel = img_path.relative_to(images_dir)
        except ValueError:
            return img_path.parent.name
        return str(rel.parent) if str(rel.parent) != "." else None

    if group_by == "stem":
        return re.sub(r"\d+$", "", img_path.stem)

    return None


def _yolo_norm_to_xyxy(xc: float, yc: float, w: float, h: float, width: int, height: int) -> BBox:
    """YOLO 归一化 (中心/宽高) -> 绝对像素 xyxy。"""
    cx, cy = xc * width, yc * height
    bw, bh = w * width, h * height
    return BBox.from_xywh(cx - bw / 2.0, cy - bh / 2.0, bw, bh)
