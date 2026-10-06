"""推理结果的数据结构与归一化。

`build_frame_result()` 是**纯函数**：输入是子进程抽出的原始 payload
（numpy 已转普通类型），输出统一的 `FrameResult`。

这样「结果长什么样」与「怎么从 ultralytics 拿出来」解耦，
前者可以脱离 torch/ultralytics 单独测试（与 `core/eval/result.py` 同理）。

坐标口径：**绝对像素 xyxy**，与统一 IR（`core/ir.py` 的 BBox）一致。
因此后续 M7 把这些结果写进 IR 时不需要二次换算（README 决策 7：
只在出口归一化一次）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .spec import TASK_CLASSIFY, TASK_DETECT, TASK_SEGMENT


def _f(value: Any) -> Optional[float]:
    """安全转 float：无法转换或 NaN 时返回 None（不编造 0）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round(v, 4)


def _i(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list:
    """把 numpy 数组 / 元组 / None 安全转成 list。

    不能用 `value or []`：numpy 数组的真值判断会抛
    `ValueError: truth value of an array ... is ambiguous`（见 R-23）。
    """
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


@dataclass
class Detection:
    """一个检测框（绝对像素 xyxy）。"""

    class_index: int
    class_name: str
    confidence: Optional[float]
    bbox: Tuple[float, float, float, float]
    # 分割掩码面积占比（仅分割任务），无则 None
    mask_area_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "mask_area_ratio": self.mask_area_ratio,
        }


@dataclass
class FrameResult:
    """一帧（或一张图）的推理结果。"""

    ok: bool = False
    error: str = ""
    # 回显请求 id，便于把响应与请求配对
    frame_id: Optional[int] = None
    # 推理耗时（毫秒），仅推理阶段
    duration_ms: Optional[float] = None
    task: str = TASK_DETECT
    # 原图尺寸，前端据此缩放叠加
    width: int = 0
    height: int = 0
    # 检测任务：框列表
    detections: List[Detection] = field(default_factory=list)
    # 分类任务：整图类别（top-1 + 可选 top-k）
    top1: Optional[Dict[str, Any]] = None
    topk: List[Dict[str, Any]] = field(default_factory=list)
    # 类名清单（下标 -> 名字），前端展示与配色用
    class_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "frame_id": self.frame_id,
            "duration_ms": self.duration_ms,
            "task": self.task,
            "width": self.width,
            "height": self.height,
            "detections": [d.to_dict() for d in self.detections],
            "top1": self.top1,
            "topk": self.topk,
            "class_names": list(self.class_names),
            "num_detections": len(self.detections),
        }


def build_frame_result(
    payload: Dict[str, Any],
    frame_id: Optional[int] = None,
    duration_ms: Optional[float] = None,
) -> FrameResult:
    """把子进程抽出的原始 payload 归一成 FrameResult。

    payload 结构（由 `ultralytics_infer` 产出）：

        {
          "task": "detect",
          "width": 640, "height": 480,
          "class_names": ["cat", "dog"],
          "detections": [
             {"class_index":0, "confidence":0.9, "bbox":[x1,y1,x2,y2], "mask_area_ratio":null},
             ...
          ],
          "top1": {"class_index":0,"class_name":"cat","confidence":0.87},   # 分类任务
          "topk": [ ... ]
        }

    缺失字段一律留空 / 中性值，绝不编造 0 或假类别。
    """
    task = str(payload.get("task") or TASK_DETECT).lower()
    names = [str(n) for n in _as_list(payload.get("class_names"))]

    result = FrameResult(
        ok=bool(payload.get("ok", True)),
        error=str(payload.get("error") or ""),
        frame_id=frame_id,
        duration_ms=duration_ms if duration_ms is not None else _f(payload.get("duration_ms")),
        task=task,
        width=_i(payload.get("width")) or 0,
        height=_i(payload.get("height")) or 0,
        class_names=names,
    )

    if task == TASK_CLASSIFY:
        top1 = payload.get("top1")
        if isinstance(top1, dict):
            result.top1 = _normalize_class_hit(top1, names)
        result.topk = [
            item
            for item in (
                _normalize_class_hit(x, names) for x in _as_list(payload.get("topk")) if isinstance(x, dict)
            )
            if item is not None
        ]
        return result

    # 检测 / 分割
    for raw in _as_list(payload.get("detections")):
        if not isinstance(raw, dict):
            continue
        det = _normalize_detection(raw, names)
        if det is not None:
            result.detections.append(det)
    return result


def _normalize_class_hit(raw: Dict[str, Any], names: List[str]) -> Optional[Dict[str, Any]]:
    """归一分类命中项（top1 / top-k 的单项）。"""
    ci = _i(raw.get("class_index"))
    if ci is None:
        return None
    name = raw.get("class_name")
    if not name:
        name = names[ci] if 0 <= ci < len(names) else f"class_{ci}"
    return {
        "class_index": ci,
        "class_name": str(name),
        "confidence": _f(raw.get("confidence")),
    }


def _normalize_detection(raw: Dict[str, Any], names: List[str]) -> Optional[Detection]:
    """归一单个检测框。缺 bbox 或坐标非法时返回 None（丢弃而不是编造）。"""
    bbox = _as_list(raw.get("bbox"))
    if len(bbox) < 4:
        return None
    coords = [_f(v) for v in bbox[:4]]
    if any(c is None for c in coords):
        return None

    ci = _i(raw.get("class_index"))
    if ci is None:
        ci = -1
    name = raw.get("class_name")
    if not name:
        name = names[ci] if 0 <= ci < len(names) else f"class_{ci}"

    return Detection(
        class_index=ci,
        class_name=str(name),
        confidence=_f(raw.get("confidence")),
        bbox=(coords[0], coords[1], coords[2], coords[3]),  # type: ignore[arg-type]
        mask_area_ratio=_f(raw.get("mask_area_ratio")),
    )


def task_of(payload_task: str) -> str:
    """归一任务名，未知时退化为 detect。"""
    t = str(payload_task or "").lower()
    return t if t in (TASK_DETECT, TASK_CLASSIFY, TASK_SEGMENT) else TASK_DETECT


__all__ = [
    "Detection",
    "FrameResult",
    "build_frame_result",
    "task_of",
]
