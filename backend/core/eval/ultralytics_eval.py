"""ultralytics 评估实现。

只在评估子进程里 import ultralytics。指标从 `model.val()` 的**返回值**取，
不 hook 内部：`results_dict` / `box` / `confusion_matrix` / `save_dir` 都是公开属性，
每一项都做了存在性判断，不同版本缺字段时退化为「没有这个指标」而不是报错。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result import BACKGROUND_LABEL, EvalResult, build_result
from .spec import EvalSpec

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _f(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    return round(v, 6)


def _plain_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): _f(v) for k, v in (d or {}).items()}


def _as_list(value: Any) -> list:
    """把 numpy 数组 / 元组 / None 安全地转成 list。

    不能用 `value or []`：numpy 数组的真值判断会抛
    `ValueError: truth value of an array ... is ambiguous`。
    """
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _list_images(save_dir: Optional[str]) -> List[str]:
    if not save_dir:
        return []
    base = Path(save_dir)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def extract_metrics(results) -> Dict[str, Any]:
    """从 val() 返回值抽出可序列化的指标。缺字段时留空，不编造 0。"""
    payload: Dict[str, Any] = {"ok": True, "error": ""}
    payload["results_dict"] = _plain_dict(getattr(results, "results_dict", None) or {})

    names = {int(k): str(v) for k, v in (getattr(results, "names", None) or {}).items()}

    # 检测/分割用 box 或 seg；分类没有逐类指标。
    # 不能用 `a or b`：Metric 对象带 __len__，nc=0 时会被判为假值。
    metric = getattr(results, "box", None)
    if metric is None:
        metric = getattr(results, "seg", None)
    nt_per_class = _as_list(getattr(results, "nt_per_class", None))

    per_class: List[Dict[str, Any]] = []
    if metric is not None:
        idxs = _as_list(getattr(metric, "ap_class_index", None))
        p = _as_list(getattr(metric, "p", None))
        r = _as_list(getattr(metric, "r", None))
        f1 = _as_list(getattr(metric, "f1", None))
        ap50 = _as_list(getattr(metric, "ap50", None))
        ap = _as_list(getattr(metric, "ap", None))

        def pick(seq: list, i: int) -> Any:
            return seq[i] if i < len(seq) else None

        for j, class_index in enumerate(idxs):
            ci = int(class_index)
            per_class.append(
                {
                    "index": ci,
                    "name": names.get(ci, f"class_{ci}"),
                    "instances": int(nt_per_class[ci]) if ci < len(nt_per_class) else 0,
                    "precision": _f(pick(p, j)),
                    "recall": _f(pick(r, j)),
                    "f1": _f(pick(f1, j)),
                    "ap50": _f(pick(ap50, j)),
                    "ap50_95": _f(pick(ap, j)),
                }
            )
    payload["per_class"] = per_class

    speed = getattr(results, "speed", None)
    payload["speed"] = {str(k): _f(v) for k, v in (speed or {}).items()} if speed else {}

    # 混淆矩阵（行=真实，列=预测，含 background）
    cm = getattr(results, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None)
    if matrix is not None:
        nc = int(getattr(cm, "nc", len(names)) or len(names))
        labels = [BACKGROUND_LABEL] + [names.get(i, f"class_{i}") for i in range(nc)]
        try:
            payload["confusion_matrix"] = {
                "labels": labels,
                "matrix": [[int(v) for v in row] for row in matrix.tolist()],
            }
        except Exception:
            payload["confusion_matrix"] = None

    save_dir = getattr(results, "save_dir", None)
    payload["save_dir"] = str(save_dir) if save_dir else ""
    payload["artifacts"] = _list_images(payload["save_dir"])
    return payload


def run(spec: EvalSpec) -> int:
    """在评估子进程中执行 ultralytics 验证。返回退出码。"""
    from ultralytics import YOLO  # 延迟导入：只发生在评估子进程里

    project = str(Path(spec.project).expanduser().resolve())
    model = YOLO(spec.weights)

    kwargs: Dict[str, Any] = {
        "data": spec.data_yaml,
        "split": spec.split,
        "imgsz": spec.imgsz,
        "batch": spec.batch,
        "device": spec.device,
        "workers": spec.workers,
        "conf": spec.conf,
        "iou": spec.iou,
        "project": project,
        "name": spec.name,
        "exist_ok": True,
        "verbose": True,
    }
    kwargs.update(spec.extra)

    started = time.time()
    try:
        results = model.val(**kwargs)
    except Exception as exc:
        failed = EvalResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            split=spec.split,
            task=spec.task,
            weights=spec.weights,
            data_yaml=spec.data_yaml,
            job_id=spec.job_id,
            tag=spec.tag,
            eval_id=spec.name,
            model_name=Path(spec.weights).name,
            created_at=datetime.now().isoformat(timespec="seconds"),
            duration_sec=round(time.time() - started, 3),
        )
        failed.save(spec.result_path)
        raise

    payload = extract_metrics(results)
    result = build_result(payload, spec, eval_id=spec.name, duration_sec=time.time() - started)
    result.save(spec.result_path)
    return 0


__all__ = ["extract_metrics", "run"]
