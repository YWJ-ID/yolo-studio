"""M6 实时验证：推理会话单例与权重来源。

推理会话是**常驻**的（见 core/infer/session.py 的说明），因此这里维护一个
全局单例；同时用一把锁保证同一时刻只有一个请求在向子进程发帧
（会话本身不是线程安全的）。

放在服务层而不是路由里：路由只做 HTTP 与参数校验，会话归属属于服务层。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.infer import (
    InferError,
    InferSession,
    InferSpec,
    capability_report,
    detect_weights_format,
)

from .config import settings

_lock = threading.RLock()
_session: Optional[InferSession] = None
# 串行化对会话的访问：同一时刻只允许一个帧在推理
_op_lock = threading.Lock()


def get_session() -> InferSession:
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                settings.ensure_dirs()
                _session = InferSession(python=str(settings.python_executable))
    return _session


def set_session(session: Optional[InferSession]) -> None:
    """替换单例。供测试注入受控会话。"""
    global _session
    with _lock:
        _session = session


def reset_session() -> None:
    global _session
    with _lock:
        old = _session
        _session = None
    if old is not None:
        try:
            old.stop()
        except Exception:
            pass


def session_lock() -> threading.Lock:
    """返回串行化锁。调用方需用它包住一帧的发送与接收。"""
    return _op_lock


def formats_payload() -> Dict[str, Any]:
    """格式清单 + 当前会话状态。"""
    session = get_session()
    return {
        "formats": capability_report(),
        "active": session.snapshot(),
    }


def load_payload(req) -> Dict[str, Any]:
    """按请求加载权重。返回会话快照；失败抛 InferError。"""
    weights = str(req.weights or "").strip()
    if not weights:
        raise InferError("weights 不能为空")

    fmt = str(getattr(req, "fmt", "") or "")
    if not fmt:
        try:
            fmt = detect_weights_format(weights)
        except ValueError as exc:
            raise InferError(str(exc)) from exc

    spec = InferSpec(
        weights=weights,
        name=str(getattr(req, "name", "") or ""),
        task=str(getattr(req, "task", "") or ""),
        classes=[str(c) for c in (getattr(req, "classes", None) or [])],
        device=str(getattr(req, "device", "") or "cpu"),
        imgsz=int(getattr(req, "imgsz", 640) or 640),
        half=bool(getattr(req, "half", False)),
        fmt=fmt,
        source="api",
    )

    session = get_session()
    with _op_lock:
        session.load(spec)
        return session.snapshot()


def discover_weights() -> List[Dict[str, Any]]:
    """列出可选的权重来源：模型库里已注册的 + 已导出的产物。

    外部任意路径不在这里列（用户自己填），因此这个列表只是「方便选项」。
    """
    out: List[Dict[str, Any]] = []

    # 1. 模型库
    try:
        from core.registry import ModelRegistry

        registry = ModelRegistry(settings.models_dir)
        for card in registry.list():
            weights = (card.weights or {}).get("best") or ""
            if not weights:
                continue
            out.append(
                {
                    "value": weights,
                    "label": f"{card.name or card.model_id}（模型库）",
                    "source": "model_library",
                    "model_id": card.model_id,
                    "task": card.task,
                    "classes": list(card.classes or []),
                    "format": "pt",
                    "exists": Path(weights).is_file(),
                }
            )
    except Exception:
        pass

    # 2. 已导出的产物
    try:
        acquires = _deploy_artifacts()
        out.extend(acquires)
    except Exception:
        pass

    # 去重（同一路径只留一条）
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in out:
        key = str(item["value"]).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _deploy_artifacts() -> List[Dict[str, Any]]:
    """扫描导出目录，把可作为推理输入的产物列出来。"""
    items: List[Dict[str, Any]] = []
    root = Path(settings.deploys_dir)
    if not root.is_dir():
        return items
    for result_file in sorted(root.glob("*/deploy_result.json")):
        import json

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for art in data.get("artifacts") or []:
            if not art.get("ok"):
                continue
            path = str(art.get("path") or "")
            if not path or not Path(path).exists():
                continue
            try:
                fmt = detect_weights_format(path)
            except ValueError:
                continue
            items.append(
                {
                    "value": path,
                    "label": f"{Path(path).name}（导出产物 · {art.get('format', '')}）",
                    "source": "deploy",
                    "model_id": str(data.get("model_id") or ""),
                    "task": "",
                    "classes": [],
                    "format": fmt,
                    "exists": True,
                }
            )
    return items


__all__ = [
    "discover_weights",
    "formats_payload",
    "get_session",
    "load_payload",
    "reset_session",
    "session_lock",
    "set_session",
]
