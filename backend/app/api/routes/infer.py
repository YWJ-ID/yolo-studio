"""M6 实时验证接口。

与前几个模块的 WebSocket 不同：这里不是「服务端单向推」，
而是**浏览器持续发帧、服务端回结果**的双向高频通信，因此要额外处理：

1. **不阻塞事件循环**：推理是同步阻塞的，用 `asyncio.to_thread` 丢到线程池里跑；
2. **背压 / 丢帧**：客户端发得比推理快时，**丢弃未处理的旧帧**而不是排队，
   否则延迟会越积越大（排队会让画面越来越滞后于现实）。
   具体做法：同一时刻只保留「最新一帧」等处理，中间到达的直接标记为丢弃并回执。
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from core.infer import InferError, InferOptions

from ...schemas import (
    InferCloseResponse,
    InferFormatsResponse,
    InferImageRequest,
    InferImageResponse,
    InferLoadRequest,
    InferSessionResponse,
    InferWeightsResponse,
)
from ...services_infer import (
    discover_weights,
    formats_payload,
    get_session,
    load_payload,
    reset_session,
    session_lock,
)

router = APIRouter(prefix="/api/infer", tags=["infer"])


def _guard(fn, *args, **kwargs):
    """把 core 的 InferError 翻译成 HTTP 400。"""
    try:
        return fn(*args, **kwargs)
    except InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


@router.get("/formats", response_model=InferFormatsResponse)
def list_formats() -> Dict[str, Any]:
    """可推理格式及其本机可用性 + 当前会话状态。"""
    return formats_payload()


@router.get("/weights", response_model=InferWeightsResponse)
def list_weights() -> Dict[str, Any]:
    """可选权重来源（模型库 + 导出产物）。外部路径由用户自行填写。"""
    return {"weights": discover_weights()}


@router.get("/session", response_model=InferSessionResponse)
def get_session_state() -> Dict[str, Any]:
    """当前推理会话状态。"""
    return {"ok": True, "session": get_session().snapshot()}


@router.post("/models", response_model=InferSessionResponse)
def load_model(req: InferLoadRequest) -> Dict[str, Any]:
    """加载权重（同步阻塞，首次要 import torch，可能几秒）。"""
    session = _guard(load_payload, req)
    return {"ok": True, "session": session}


@router.post("/close", response_model=InferCloseResponse)
def close_session() -> Dict[str, Any]:
    """关闭推理会话，释放子进程。"""
    reset_session()
    return {"ok": True, "closed": True}


@router.post("/image", response_model=InferImageResponse)
def infer_image(req: InferImageRequest) -> Dict[str, Any]:
    """单张图片推理：给服务器本地 path，或 base64 image。

    用于「上传图片验证」以及前端的一次性快照。摄像头流走 WebSocket。
    """
    session = get_session()
    if not session.alive or not session.info:
        raise HTTPException(status_code=400, detail="尚未加载模型，请先调用 POST /api/infer/models")

    options = InferOptions(
        conf=req.conf,
        iou=req.iou,
        max_det=req.max_det,
        only_classes=list(req.only_classes or []),
    )
    problems = options.validate()
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    with session_lock():
        try:
            if req.image:
                raw = req.image.split(",", 1)[-1]  # 容忍 data:image/...;base64, 前缀
                base64.b64decode(raw)  # 提前校验是否合法 base64
                result = session.infer_jpeg(raw, options)
            elif req.path:
                from ...config import settings

                if not settings.is_path_allowed(req.path):
                    raise HTTPException(status_code=403, detail="该路径不在允许读取的范围内")
                if not __import__("pathlib").Path(req.path).is_file():
                    raise HTTPException(status_code=404, detail=f"图片不存在: {req.path}")
                result = session.infer_path(req.path, options)
            else:
                raise HTTPException(status_code=400, detail="需要提供 path 或 image 之一")
        except InferError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "result": result.to_dict()}


# ---------------------------------------------------------------------------
# WebSocket：实时流
# ---------------------------------------------------------------------------


@router.websocket("/ws")
async def infer_ws(websocket: WebSocket) -> None:
    """实时推理流。

    客户端消息：
        {"type":"load","spec":{...InferSpec...}}     # 也可先走 REST 加载
        {"type":"frame","data":"<base64 jpeg>","options":{...}}
        {"type":"close"}

    服务端消息：
        {"type":"ready","session":{...}}
        {"type":"loaded","ok":true,"session":{...}}
        {"type":"result","id":N,"result":{...}}      # 推理结果
        {"type":"dropped","id":N}                    # 因背压被丢弃的帧
        {"type":"error","error":".."}
    """
    await websocket.accept()
    session = get_session()

    await websocket.send_json({"type": "ready", "session": session.snapshot()})

    # 背压：只保留最新一帧待处理。正在推理时新到的帧覆盖 pending。
    pending: Optional[Dict[str, Any]] = None
    pending_lock = asyncio.Lock()
    processing = False

    async def run_frame(msg: Dict[str, Any]) -> None:
        """在线程池里跑一帧，把结果发回。"""
        frame_id = msg.get("id")
        options = InferOptions.from_dict(msg.get("options"))
        raw = str(msg.get("data") or "").split(",", 1)[-1]
        try:
            with session_lock():
                result = await asyncio.to_thread(session.infer_jpeg, raw, options, frame_id)
            await websocket.send_json(
                {"type": "result", "id": frame_id, "result": result.to_dict()}
            )
        except InferError as exc:
            await websocket.send_json({"type": "error", "id": frame_id, "error": str(exc)})
        except WebSocketDisconnect:
            raise
        except Exception as exc:
            try:
                await websocket.send_json(
                    {"type": "error", "id": frame_id, "error": f"{type(exc).__name__}: {exc}"}
                )
            except Exception:
                pass

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "load":
                try:
                    with session_lock():
                        info = await asyncio.to_thread(session.load, _spec_from_msg(msg))
                    await websocket.send_json(
                        {"type": "loaded", "ok": True, "session": session.snapshot(), "info": info}
                    )
                except InferError as exc:
                    await websocket.send_json({"type": "loaded", "ok": False, "error": str(exc)})
                continue

            if mtype == "close":
                await websocket.send_json({"type": "closed"})
                break

            if mtype == "frame":
                if not session.alive or not session.info:
                    await websocket.send_json(
                        {"type": "error", "id": msg.get("id"), "error": "尚未加载模型"}
                    )
                    continue

                async with pending_lock:
                    if processing:
                        # 已有帧在推理：覆盖 pending（丢弃中间帧）
                        dropped_id = pending.get("id") if pending else None
                        pending = msg
                        if dropped_id is not None:
                            await websocket.send_json({"type": "dropped", "id": dropped_id})
                        continue
                    processing = True

                try:
                    await run_frame(msg)
                finally:
                    # 处理期间可能又来了新帧，取最新的一帧继续
                    while True:
                        async with pending_lock:
                            nxt = pending
                            pending = None
                            if nxt is None:
                                processing = False
                                break
                        await run_frame(nxt)
                continue

            await websocket.send_json({"type": "error", "error": f"未知指令: {mtype}"})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def _spec_from_msg(msg: Dict[str, Any]):
    """从 WS 的 load 消息构造 InferSpec。"""
    from core.infer import InferSpec, detect_weights_format

    spec_dict = dict(msg.get("spec") or msg)
    spec_dict.pop("type", None)
    weights = str(spec_dict.get("weights") or "")
    if not spec_dict.get("fmt") and weights:
        try:
            spec_dict["fmt"] = detect_weights_format(weights)
        except ValueError:
            pass
    spec_dict.setdefault("device", "cpu")
    return InferSpec.from_dict(spec_dict)
