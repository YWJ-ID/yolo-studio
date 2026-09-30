"""OpenLABEL / VCD 格式适配器（面向 DMD 驾驶员监控数据集）。

## 与 YOLO 适配器的本质差异

DMD 的标注是**逐帧的时序动作标签**，格式形如：

    "actions": {
      "1": {"type": "driver_actions/texting_right",
            "frame_intervals": [{"frame_start": 10, "frame_end": 29}]}
    }

它**没有边界框**，对应的是 YOLO 的**分类任务**，不是检测任务。
因此本适配器产出 `kind="image"` 的标注（见 core/ir.py），
由导出模块按分类目录结构（`{split}/{class_name}/*.jpg`）落盘。

## 输入

    <root>/
    ├── sessionA.json        # VCD 文件（含 "openlabel" 顶层键）
    └── frames/
        ├── 000000.jpg       # 文件名中的数字即帧号
        └── ...

帧号从文件名末尾的数字串提取，因此 `000059.jpg` 与 `sessionA_frame_59.jpg` 都能识别。

## 防泄漏

同一段视频的相邻帧几乎完全相同，绝不能跨 train/val。
本适配器把整段视频的帧打上同一个 `group`（取 VCD 的 name），
划分模块据此保证同一视频整体落入同一个子集。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ir import KIND_IMAGE, Annotation, Category, DatasetBundle, ImageRecord, make_uid
from ..io import is_image, probe_size
from .base import BaseAdapter, normalize_category_name

# TaTo 定义的两个特殊标签：NAN = 该帧无对应相机画面；-- = 无标注
SPECIAL_LABELS = {"NAN", "--", ""}

# 默认作为分类标签的标注层级
DEFAULT_LEVEL = "driver_actions"

# 自动探测帧目录时的候选名
_FRAMES_DIR_CANDIDATES = ("frames", "images", "frames_rgb", "rgb")

# 文件名末尾的数字串
_FRAME_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")


def extract_frame_index(stem: str) -> Optional[int]:
    """从文件名提取帧号（取最后一串连续数字）。"""
    match = _FRAME_INDEX_RE.search(stem)
    return int(match.group(1)) if match else None


class OpenLabelAdapter(BaseAdapter):
    name = "openlabel"
    display_name = "OpenLABEL / VCD (DMD)"
    extensions = (".json",)

    # ---------- 探测 ----------

    @classmethod
    def detect(cls, root) -> bool:
        return bool(cls.find_vcd_files(Path(root)))

    @classmethod
    def find_vcd_files(cls, root: Path) -> List[Path]:
        """找出目录下的 VCD 文件（含 "openlabel" 顶层键的 json）。"""
        if not root.is_dir():
            return []
        found: List[Path] = []
        for p in sorted(root.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    head = json.load(f)
                if isinstance(head, dict) and "openlabel" in head:
                    found.append(p)
            except Exception:
                continue
        return found

    # ---------- 读取 ----------

    @classmethod
    def load(
        cls,
        root,
        source_id: Optional[str] = None,
        level: Optional[str] = DEFAULT_LEVEL,
        frames_dir: Optional[str] = None,
        include_objects: bool = False,
        **options,
    ) -> DatasetBundle:
        root = Path(root).resolve()
        source_id = source_id or f"openlabel:{root.name}"

        bundle = DatasetBundle(source_id=source_id, format_name=cls.name, root=str(root))
        bundle.meta["level"] = level
        bundle.meta["include_objects"] = include_objects

        vcd_files = cls.find_vcd_files(root)
        if not vcd_files:
            bundle.warnings.append(f"未找到 VCD 文件（含 openlabel 键的 json）: {root}")
            return bundle

        resolved_frames_dir = cls._resolve_frames_dir(root, frames_dir, vcd_files)
        if resolved_frames_dir is None:
            bundle.warnings.append(
                f"未找到帧图片目录，请用 frames_dir 指定。已在 {root} 下查找 "
                f"{list(_FRAMES_DIR_CANDIDATES)}"
            )
            return bundle
        bundle.meta["frames_dir"] = str(resolved_frames_dir)

        for vcd_path in vcd_files:
            cls._load_one_vcd(
                bundle=bundle,
                vcd_path=vcd_path,
                frames_dir=resolved_frames_dir,
                source_id=source_id,
                level=level,
                include_objects=include_objects,
            )

        return bundle

    # ---------- 内部 ----------

    @classmethod
    def _load_one_vcd(
        cls,
        bundle: DatasetBundle,
        vcd_path: Path,
        frames_dir: Path,
        source_id: str,
        level: Optional[str],
        include_objects: bool,
    ) -> None:
        with open(vcd_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        openlabel = data.get("openlabel") or {}
        video_name = cls._video_name(openlabel, vcd_path)
        total_frames = cls._total_frames(openlabel)

        # 1) 收集 (帧区间, 类别) 列表
        intervals: List[Tuple[int, int, str]] = []
        intervals += cls._action_intervals(openlabel, level)
        if include_objects:
            intervals += cls._object_intervals(openlabel)

        if not intervals:
            bundle.warnings.append(
                f"{vcd_path.name}: 未找到匹配的标注"
                f"（level={level!r}, include_objects={include_objects}）"
            )

        # 登记类别（保留原始形态，映射交给 taxonomy 模块）
        for _, _, name in intervals:
            bundle.add_category(Category(name=name, original_name=name, meta={"source": vcd_path.name}))

        # 2) 扫描帧图片，建立 帧号 -> 路径
        frame_images = cls._scan_frame_images(frames_dir)
        if not frame_images:
            bundle.warnings.append(f"帧目录中没有图像: {frames_dir}")
            return

        # 3) 只为实际存在的帧图建立标注（避免为不存在的帧生成空标注）
        label_by_frame: Dict[int, List[str]] = {}
        for start, end, name in intervals:
            for idx in frame_images:
                if start <= idx <= end:
                    label_by_frame.setdefault(idx, []).append(name)

        # 4) 生成图像记录 + 图像级标注
        for idx in sorted(frame_images):
            img_path = frame_images[idx]
            rel = str(img_path.relative_to(frames_dir))

            image = cls.build_image(
                path=img_path,
                source_id=source_id,
                root=frames_dir,
                split=None,          # 交由划分模块决定
                group=video_name,    # 整段视频同组，防止跨集合泄漏
                probe_size_fn=probe_size,
            )
            image.meta["frame_index"] = idx
            image.meta["video"] = video_name
            bundle.add_image(image)

            labels = label_by_frame.get(idx)
            if not labels:
                bundle.warnings.append(
                    f"帧 {idx} 无标注（{rel}），可能是标注空洞或 level 选择有误"
                )
                continue
            if len(labels) > 1:
                bundle.warnings.append(f"帧 {idx} 命中多个类别 {labels}，已全部保留")
            for name in labels:
                bundle.add_annotation(
                    Annotation(
                        image_uid=image.uid,
                        category=name,
                        bbox=None,
                        kind=KIND_IMAGE,
                        group=video_name,
                        meta={"video": video_name, "frame_index": idx, "level": level},
                    )
                )

        # 5) 帧号越界检查
        if total_frames is not None:
            outside = [i for i in frame_images if i >= total_frames or i < 0]
            if outside:
                bundle.warnings.append(
                    f"{vcd_path.name}: {len(outside)} 个帧号超出 VCD 声明范围 "
                    f"(total_frames={total_frames})，例如 {sorted(outside)[:5]}"
                )

        bundle.meta.setdefault("videos", []).append(
            {
                "vcd": vcd_path.name,
                "name": video_name,
                "total_frames": total_frames,
                "frames_on_disk": len(frame_images),
            }
        )

    # ---------- 解析辅助 ----------

    @staticmethod
    def _video_name(openlabel: Dict[str, Any], vcd_path: Path) -> str:
        meta = openlabel.get("metadata") or {}
        return str(meta.get("name") or vcd_path.stem)

    @staticmethod
    def _total_frames(openlabel: Dict[str, Any]) -> Optional[int]:
        streams = openlabel.get("streams") or {}
        best: Optional[int] = None
        for stream in streams.values():
            props = (stream or {}).get("stream_properties") or {}
            total = props.get("total_frames")
            if isinstance(total, int):
                best = total if best is None else max(best, total)
        return best

    @classmethod
    def _action_intervals(
        cls, openlabel: Dict[str, Any], level: Optional[str]
    ) -> List[Tuple[int, int, str]]:
        """从 actions 提取 (start, end, 类别名)。

        action 的 type 形如 "driver_actions/texting_right"。
        level 给定时只取该层级，并且只保留斜杠后的标签名；
        level 为 None 时保留完整 type 作为类别名。
        """
        out: List[Tuple[int, int, str]] = []
        actions = openlabel.get("actions") or {}
        for action in actions.values():
            semantic = str((action or {}).get("type") or "")
            if not semantic:
                continue

            if level:
                prefix = f"{level}/"
                if not semantic.startswith(prefix):
                    continue
                category = semantic[len(prefix):]
            else:
                category = semantic

            category = normalize_category_name(category)
            if category in SPECIAL_LABELS:
                continue

            for start, end in cls._frame_ranges(action):
                out.append((start, end, category))
        return out

    @classmethod
    def _object_intervals(cls, openlabel: Dict[str, Any]) -> List[Tuple[int, int, str]]:
        """从 objects 提取 (start, end, 类别名)。object 的 type 就是标签名。"""
        out: List[Tuple[int, int, str]] = []
        objects = openlabel.get("objects") or {}
        for obj in objects.values():
            category = normalize_category_name(str((obj or {}).get("type") or ""))
            if category in SPECIAL_LABELS:
                continue
            for start, end in cls._frame_ranges(obj):
                out.append((start, end, category))
        return out

    @staticmethod
    def _frame_ranges(element: Dict[str, Any]) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        for fi in (element or {}).get("frame_intervals") or []:
            try:
                ranges.append((int(fi["frame_start"]), int(fi["frame_end"])))
            except (KeyError, TypeError, ValueError):
                continue
        return ranges

    @staticmethod
    def _resolve_frames_dir(
        root: Path, frames_dir: Optional[str], vcd_files: List[Path]
    ) -> Optional[Path]:
        if frames_dir:
            candidate = (root / frames_dir) if not Path(frames_dir).is_absolute() else Path(frames_dir)
            return candidate.resolve() if candidate.is_dir() else None

        for name in _FRAMES_DIR_CANDIDATES:
            candidate = root / name
            if candidate.is_dir():
                return candidate

        # 退化：VCD 同目录下直接放图片
        if any(is_image(p) for p in root.glob("*")):
            return root
        return None

    @staticmethod
    def _scan_frame_images(frames_dir: Path) -> Dict[int, Path]:
        """扫描帧图，返回 {帧号: 路径}；帧号无法解析的会被跳过。"""
        result: Dict[int, Path] = {}
        for p in sorted(frames_dir.rglob("*")):
            if not p.is_file() or not is_image(p):
                continue
            idx = extract_frame_index(p.stem)
            if idx is None:
                continue
            result[idx] = p
        return result
