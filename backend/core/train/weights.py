"""训练基础模型的候选发现与上传校验。

训练时 `weights` 可以写四种东西（解析规则见 `UltralyticsBackend.resolve_weights`）：

  * ultralytics 结构文件 `.yaml`（从零训练，**离线可用**，不下载）；
  * 预训练权重 `.pt`（放在 `YOLO_STUDIO_WEIGHTS` 目录里按文件名引用）；
  * 任意存在的绝对路径（例如历史训练产物、模型库里的 best.pt）；
  * 交给 ultralytics 自动下载的名字（如 `yolo11n.pt`，首次使用会联网）。

本模块只回答两个问题：
  1. 「有哪些基础模型可以直接选」（`candidate_weights`）；
  2. 「上传的权重文件名是否合法」（`normalize_weight_filename`）。

它不 import torch/ultralytics，也不读全局配置（目录与模型卡片都由调用方传入），
因此可在 API 进程里安全调用，也便于单独测试。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .spec import TASK_CLASSIFY, TASK_DETECT

# 权重形态
FORMAT_STRUCTURE = "structure"
FORMAT_PT = "pt"

# 允许作为基础模型上传的后缀
WEIGHT_SUFFIXES = (".pt", ".yaml", ".yml")

# 上传体积上限（.pt 可能很大，给一个足够宽松但能挡住误传大文件的上限）
MAX_WEIGHT_BYTES = 4 * 1024 ** 3

# 内置结构文件：ultralytics 包自带，列出即表示「可离线从零训练」
STRUCTURE_WEIGHTS: Dict[str, List[str]] = {
    TASK_DETECT: ["yolo11n.yaml", "yolo11s.yaml", "yolo11m.yaml", "yolo11l.yaml", "yolo11x.yaml"],
    TASK_CLASSIFY: ["yolo11-cls.yaml"],
}

# 内置预训练权重：本地权重目录里没有时会由 ultralytics 联网下载
PRETRAINED_WEIGHTS: Dict[str, List[str]] = {
    TASK_DETECT: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
    TASK_CLASSIFY: ["yolo11n-cls.pt"],
}

# 结果排序：内置在前，历史产物在后
SOURCE_ORDER = {"builtin": 0, "weights_dir": 1, "model_library": 2, "run": 3}


def weight_format(name: str) -> str:
    """按扩展名判断权重形态；不是受支持的权重返回空串。"""
    low = str(name or "").lower()
    if low.endswith((".yaml", ".yml")):
        return FORMAT_STRUCTURE
    if low.endswith(".pt"):
        return FORMAT_PT
    return ""


def normalize_weight_filename(name: str) -> str:
    """从上传的文件名里取出安全的文件名并校验后缀。

    只取最后一段（防目录穿越），拒绝空名与 `..`，只接受 .pt/.yaml/.yml。
    非法时抛 ValueError（调用方翻译成 HTTP 400）。
    """
    raw = str(name or "").replace("\\", "/").strip()
    base = raw.rsplit("/", 1)[-1].strip()
    if not base or base in (".", "..") or ".." in base:
        raise ValueError("权重文件名无效")
    if not weight_format(base):
        raise ValueError("只支持 .pt / .yaml / .yml 权重文件")
    return base


def _entry(value: str, label: str, source: str, **extra: Any) -> Dict[str, Any]:
    """拼一条权重候选项。字段与 app.schemas.BaseWeightItem 对齐。"""
    item: Dict[str, Any] = {
        "value": value,
        "label": label,
        "source": source,
        "task": "",
        "classes": [],
        "format": weight_format(value),
        "exists": True,
        "size_bytes": 0,
    }
    item.update(extra)
    return item


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def builtin_weights(weights_dir: Optional[Path] = None, tasks: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """内置结构文件与预训练权重。

    预训练 `.pt` 若已在权重目录里则标记 exists=True（离线可用），否则提示「需联网下载」。
    """
    root = Path(weights_dir) if weights_dir else None
    items: List[Dict[str, Any]] = []
    for task in tasks or list(STRUCTURE_WEIGHTS.keys()):
        for name in STRUCTURE_WEIGHTS.get(task, []):
            items.append(
                _entry(name, f"{name}（结构文件 · 从零训练）", "builtin", task=task, format=FORMAT_STRUCTURE, exists=False)
            )
        for name in PRETRAINED_WEIGHTS.get(task, []):
            local = (root / name) if root else None
            exists = bool(local and local.is_file())
            hint = "已缓存" if exists else "需联网下载"
            items.append(
                _entry(
                    name,
                    f"{name}（预训练 · {hint}）",
                    "builtin",
                    task=task,
                    format=FORMAT_PT,
                    exists=exists,
                    size_bytes=_size_of(local) if exists and local else 0,
                )
            )
    return items


def scan_weights_dir(weights_dir: Optional[Path], skip_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """扫描权重目录，列出可直接引用的 .pt/.yaml 文件。"""
    root = Path(weights_dir) if weights_dir else None
    if not root or not root.is_dir():
        return []
    skip = {n.lower() for n in (skip_names or set())}
    items: List[Dict[str, Any]] = []
    for p in sorted(root.iterdir()):
        if not p.is_file() or p.name.lower() in skip:
            continue
        fmt = weight_format(p.name)
        if not fmt:
            continue
        items.append(
            _entry(str(p.resolve()), f"{p.name}（权重目录）", "weights_dir", format=fmt, exists=True, size_bytes=_size_of(p))
        )
    return items


def _task_from_job(run_dir: Path) -> str:
    """从训练目录的 job.json 里读回 task，用于按任务类型筛选候选。"""
    job_file = run_dir / "job.json"
    if not job_file.is_file():
        return ""
    try:
        data = json.loads(job_file.read_text(encoding="utf-8"))
    except Exception:
        return ""
    spec = data.get("spec") if isinstance(data, dict) else None
    return str((spec or {}).get("task") or "")


def scan_runs(runs_dir: Optional[Path], limit: int = 20) -> List[Dict[str, Any]]:
    """扫描历史训练产物，列出可继续微调的 best.pt / last.pt（最新的在前）。"""
    root = Path(runs_dir) if runs_dir else None
    if not root or not root.is_dir():
        return []
    run_dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    items: List[Dict[str, Any]] = []
    for run in run_dirs[: max(0, int(limit))]:
        wdir = run / "weights"
        if not wdir.is_dir():
            continue
        task = _task_from_job(run)
        for fname in ("best.pt", "last.pt"):
            p = wdir / fname
            if not p.is_file():
                continue
            items.append(
                _entry(
                    str(p.resolve()),
                    f"{run.name}/{fname}（历史训练产物）",
                    "run",
                    task=task,
                    format=FORMAT_PT,
                    exists=True,
                    size_bytes=_size_of(p),
                )
            )
    return items


def model_weights(cards: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """模型库里已注册的权重（best 优先，退而 last）。

    `cards` 可以是 `ModelCard` 对象或它的 `to_dict()` 结果。
    """
    items: List[Dict[str, Any]] = []
    for card in cards or []:
        if isinstance(card, dict):
            weights = card.get("weights") or {}
            name = card.get("name") or card.get("model_id") or ""
            model_id = card.get("model_id") or ""
            task = card.get("task") or ""
            classes = list(card.get("classes") or [])
        else:
            weights = getattr(card, "weights", None) or {}
            name = getattr(card, "name", "") or getattr(card, "model_id", "") or ""
            model_id = getattr(card, "model_id", "") or ""
            task = getattr(card, "task", "") or ""
            classes = list(getattr(card, "classes", None) or [])
        best = weights.get("best") or weights.get("last") or ""
        if not best:
            continue
        path = Path(best)
        items.append(
            _entry(
                str(best),
                f"{name}（模型库）",
                "model_library",
                task=task,
                classes=classes,
                format=FORMAT_PT,
                exists=path.is_file(),
                size_bytes=_size_of(path),
                model_id=model_id,
            )
        )
    return items


def candidate_weights(
    weights_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    cards: Optional[Iterable[Any]] = None,
    tasks: Optional[Iterable[str]] = None,
    run_limit: int = 20,
) -> List[Dict[str, Any]]:
    """汇总所有可选的基础模型来源，去重后按来源排序返回。

    去重键：文件路径按解析后的绝对路径（小写）；内置名按 `builtin:<name>`。
    权重目录里与内置同名的文件会被跳过，避免同一个模型出现两条。
    """
    builtins = builtin_weights(weights_dir, tasks)
    builtin_names = {Path(it["value"]).name.lower() for it in builtins if it["format"] == FORMAT_PT}

    ordered: List[Dict[str, Any]] = []
    ordered.extend(builtins)
    ordered.extend(model_weights(cards))
    ordered.extend(scan_weights_dir(weights_dir, skip_names=builtin_names))
    ordered.extend(scan_runs(runs_dir, limit=run_limit))

    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    for item in ordered:
        value = str(item["value"])
        if item["source"] == "builtin":
            key = f"builtin:{value.lower()}"
        else:
            try:
                key = str(Path(value).resolve()).lower()
            except Exception:
                key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique.sort(key=lambda it: (SOURCE_ORDER.get(it["source"], 9), it["label"]))
    return unique


__all__ = [
    "FORMAT_PT",
    "FORMAT_STRUCTURE",
    "MAX_WEIGHT_BYTES",
    "PRETRAINED_WEIGHTS",
    "SOURCE_ORDER",
    "STRUCTURE_WEIGHTS",
    "WEIGHT_SUFFIXES",
    "builtin_weights",
    "candidate_weights",
    "model_weights",
    "normalize_weight_filename",
    "scan_runs",
    "scan_weights_dir",
    "weight_format",
]
