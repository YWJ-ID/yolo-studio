"""ultralytics 推理实现。

**这是唯一 import ultralytics / torch 的地方**，且只在推理子进程里被调用。
API 进程绝不 import 本模块（与训练/评估/导出同一条隔离原则）。

设计要点：
- 结果统一抽成**普通 Python 类型**（numpy 转 list）后再交给上层，
  避免 numpy 数组跨进程序列化问题，也避免 R-23 那种真值判断陷阱。
- 坐标为**绝对像素 xyxy**，与统一 IR 一致。
- 缺失字段留空，不编造 0。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .spec import InferOptions, InferSpec, TASK_CLASSIFY, TASK_DETECT, TASK_SEGMENT


def _as_list(value: Any) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _names_dict(model) -> Dict[int, str]:
    """从模型拿 {下标: 类名}。ultralytics 的 model.names 通常是 dict，也可能是 list。"""
    names = getattr(model, "names", None)
    out: Dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            out[i] = str(v)
    return out


def resolve_names(model, spec: InferSpec) -> Tuple[Dict[int, str], str]:
    """解析「这次会话该用哪些类名」，返回 (names, 来源)。

    优先级：**spec.classes（显式）> 模型自带 names**。
    显式类名覆盖模型自带的：按下标逐项覆盖，模型判定类别数以外的项也登记。

    **必须由 describe_model 与 predict_image 共用**——否则会出现
    「加载时显示中文类名、推理框上却还是模型原类名」的不一致
    （这是实际踩到过的 bug：predict_image 曾自己再读一次模型 names）。
    """
    model_names = _names_dict(model)
    if spec.classes:
        names = dict(model_names)
        for i, c in enumerate(spec.classes):
            names[i] = str(c)
        return names, "request"
    return model_names, ("model" if model_names else "")


def load_model(spec: InferSpec):
    """加载权重，返回 ultralytics YOLO 对象。仅能在推理子进程中调用。"""
    from ultralytics import YOLO  # 延迟导入

    model = YOLO(spec.weights, task=spec.normalized_task() or None)
    return model


def describe_model(model, spec: InferSpec) -> Dict[str, Any]:
    """加载完成后回给主进程的描述信息。

    类名的取值优先级与推理时**完全一致**（共用 `resolve_names`），
    免得出现「加载时一个类名、框上是另一个」。
    """
    names, source = resolve_names(model, spec)
    task = spec.normalized_task() or str(getattr(model, "task", "") or TASK_DETECT)

    return {
        "task": task,
        "classes": [names[i] for i in sorted(names)],
        "num_classes": len(names),
        # 类名来源：request = 请求显式指定，model = 模型自带，空 = 都没有（会退化成 class_N）
        "class_source": source,
        "weights": spec.weights,
        "format": spec.fmt,
        "imgsz": spec.imgsz,
        "device": spec.device,
    }


def _extract_detections(result, names: Dict[int, str], task: str) -> List[Dict[str, Any]]:
    """从单个 ultralytics 结果抽检测框（绝对像素 xyxy）。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    out: List[Dict[str, Any]] = []
    xyxy = _as_list(getattr(boxes, "xyxy", None))
    clss = _as_list(getattr(boxes, "cls", None))
    confs = _as_list(getattr(boxes, "conf", None))

    # 分割掩码面积占比（可选）
    ratios: List[Optional[float]] = []
    if task == TASK_SEGMENT:
        masks = getattr(result, "masks", None)
        data = getattr(masks, "data", None) if masks is not None else None
        if data is not None:
            try:
                for m in data:
                    ratios.append(round(float(m.float().mean().item()), 6))
            except Exception:
                ratios = []

    for i, box in enumerate(xyxy):
        coords = _as_list(box)
        if len(coords) < 4:
            continue
        ci = int(_as_list(clss)[i]) if i < len(clss) else -1
        conf = float(_as_list(confs)[i]) if i < len(confs) else None
        out.append(
            {
                "class_index": ci,
                "class_name": names.get(ci, f"class_{ci}"),
                "confidence": round(conf, 6) if conf is not None else None,
                "bbox": [round(float(v), 4) for v in coords[:4]],
                "mask_area_ratio": ratios[i] if i < len(ratios) else None,
            }
        )
    return out


def _extract_classification(result, names: Dict[int, str]) -> Dict[str, Any]:
    """从分类结果抽 top-1 / top-k。"""
    probs = getattr(result, "probs", None)
    if probs is None:
        return {"top1": None, "topk": []}

    top1_idx = getattr(probs, "top1", None)
    top1_conf = getattr(probs, "top1conf", None)
    top1 = None
    if top1_idx is not None:
        ci = int(top1_idx)
        top1 = {
            "class_index": ci,
            "class_name": names.get(ci, f"class_{ci}"),
            "confidence": round(float(top1_conf), 6) if top1_conf is not None else None,
        }

    topk: List[Dict[str, Any]] = []
    data = getattr(probs, "data", None)
    if data is not None:
        try:
            flat = _as_list(data)
            pairs = [(i, float(c)) for i, c in enumerate(flat)]
            pairs.sort(key=lambda p: p[1], reverse=True)
            for ci, conf in pairs[:5]:
                topk.append(
                    {
                        "class_index": ci,
                        "class_name": names.get(ci, f"class_{ci}"),
                        "confidence": round(conf, 6),
                    }
                )
        except Exception:
            topk = []

    return {"top1": top1, "topk": topk}


def predict_image(model, source: Any, spec: InferSpec, options: InferOptions) -> Dict[str, Any]:
    """对单张图（或 numpy BGR 图）推理，返回归一前的原始 payload。"""
    kwargs: Dict[str, Any] = {
        "imgsz": spec.imgsz,
        "conf": options.conf,
        "iou": options.iou,
        "max_det": options.max_det or 300,
        "device": spec.device or None,
        "verbose": False,
    }
    if spec.half:
        kwargs["half"] = True
    kwargs.update(spec.extra)

    results = model.predict(source=source, **kwargs)
    if not results:
        return {"ok": True, "task": spec.normalized_task() or TASK_DETECT,
                "width": 0, "height": 0, "class_names": [], "detections": []}

    result = results[0]
    names, _source = resolve_names(model, spec)
    task = spec.normalized_task() or str(getattr(model, "task", "") or TASK_DETECT)

    orig = getattr(result, "orig_shape", None)
    height, width = (int(orig[0]), int(orig[1])) if orig else (0, 0)

    payload: Dict[str, Any] = {
        "ok": True,
        "task": task,
        "width": width,
        "height": height,
        "class_names": [names[i] for i in sorted(names)],
    }

    if task == TASK_CLASSIFY:
        payload.update(_extract_classification(result, names))
    else:
        dets = _extract_detections(result, names, task)
        # 类别过滤（按下标）
        if options.only_classes:
            allowed = set(int(c) for c in options.only_classes)
            dets = [d for d in dets if d["class_index"] in allowed]
        payload["detections"] = dets

    return payload


__all__ = [
    "describe_model",
    "load_model",
    "predict_image",
    "resolve_names",
]
