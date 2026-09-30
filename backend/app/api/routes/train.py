"""训练任务接口（M2）。

职责边界：本文件只做「参数校验 -> 调用 core.train -> 序列化 + 事件转发」。
训练调度、指标解析、资源采样全在 `core.train`，CLI 与界面共用同一套逻辑。

事件推送：core 的 `TrainingManager.subscribe` 只接受同步回调；
WebSocket 是异步的，因此这里用 `loop.call_soon_threadsafe` 把
监控线程发出的事件投递到事件循环的队列里。core 不认识 asyncio。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from core.train import (
    JobNotFound,
    JobStateError,
    TASKS,
    TrainSpec,
    TrainingManager,
    resolve_device,
)

from ...config import settings
from ...schemas import (
    TrainBackendInfo,
    TrainBackendsResponse,
    TrainJobResponse,
    TrainJobsResponse,
    TrainRequest,
)
from ...services import get_training_manager

router = APIRouter(prefix="/api/train", tags=["train"])


def get_manager() -> TrainingManager:
    """训练管理器单例（由 app.services 持有）。"""
    return get_training_manager()


def set_manager(manager: Optional[TrainingManager]) -> None:
    """替换单例。供测试注入受控管理器。"""
    from ...services import set_managers

    set_managers(training=manager)


def shutdown_manager() -> None:
    from ...services import shutdown_managers

    shutdown_managers()


# ---------------------------------------------------------------------------
# 后端与监控能力
# ---------------------------------------------------------------------------


@router.get("/backends", response_model=TrainBackendsResponse)
def list_backends() -> TrainBackendsResponse:
    manager = get_manager()
    backend = manager.backend
    return TrainBackendsResponse(
        backends=[TrainBackendInfo(**backend.describe())],
        monitor=manager.capabilities(),
        defaults={
            "device": resolve_device(settings.train_device),
            "python": str(settings.python_executable),
            "runs_dir": str(manager.runs_dir),
            "weights_dir": str(manager.weights_dir or settings.weights_dir),
            "tasks": list(TASKS),
        },
    )


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------


def _to_spec(req: TrainRequest, manager: TrainingManager) -> TrainSpec:
    device = req.device or settings.train_device
    return TrainSpec(
        data_yaml=req.data_yaml,
        project=str(Path(manager.runs_dir).resolve()),
        name=req.name or "",
        weights=req.weights,
        task=req.task,
        epochs=req.epochs,
        imgsz=req.imgsz,
        batch=req.batch,
        device=resolve_device(device),
        workers=req.workers,
        seed=req.seed,
        patience=req.patience,
        optimizer=req.optimizer,
        lr0=req.lr0,
        extra=dict(req.extra),
    )


def _guard(func, *args, **kwargs):
    """把 core 的异常翻译成 HTTP 状态码。"""
    try:
        return func(*args, **kwargs)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs", response_model=TrainJobResponse)
def create_job(req: TrainRequest) -> TrainJobResponse:
    """创建并立即启动一个训练任务。"""
    if not Path(req.data_yaml).is_file():
        raise HTTPException(status_code=400, detail=f"data.yaml 不存在: {req.data_yaml}")
    if req.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"未知 task: {req.task}，可选 {list(TASKS)}")

    job = _guard(get_manager().start, _to_spec(req, get_manager()))
    return TrainJobResponse(job=job.to_dict())


@router.get("/jobs", response_model=TrainJobsResponse)
def list_jobs() -> TrainJobsResponse:
    manager = get_manager()
    return TrainJobsResponse(jobs=[j.to_dict() for j in manager.list()])


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    """任务详情：状态 + 指标 + 资源快照。"""
    manager = get_manager()
    job = _guard(manager.get, job_id)
    return {
        "ok": True,
        "job": job.to_dict(),
        "metrics": manager.metrics(job_id).to_dict(max_rows=2000),
        "resources": manager.resources(job_id),
        "logs": manager.logs(job_id, limit=200),
    }


@router.post("/jobs/{job_id}/stop", response_model=TrainJobResponse)
def stop_job(job_id: str) -> TrainJobResponse:
    job = _guard(get_manager().stop, job_id)
    return TrainJobResponse(job=job.to_dict())


@router.post("/jobs/{job_id}/resume", response_model=TrainJobResponse)
def resume_job(job_id: str) -> TrainJobResponse:
    job = _guard(get_manager().resume, job_id)
    return TrainJobResponse(job=job.to_dict())


# ---------------------------------------------------------------------------
# 指标 / 日志 / 资源
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/metrics")
def job_metrics(
    job_id: str,
    since_epoch: Optional[int] = Query(None, description="只返回 epoch 大于它的行（增量拉取）"),
) -> Dict[str, Any]:
    manager = get_manager()
    _guard(manager.get, job_id)
    series = manager.metrics(job_id, since_epoch=since_epoch)
    return {"ok": True, "metrics": series.to_dict()}


@router.get("/jobs/{job_id}/logs")
def job_logs(
    job_id: str,
    offset: int = Query(0, ge=0, description="起始序号（含）"),
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    manager = get_manager()
    _guard(manager.get, job_id)
    data = manager.logs(job_id, offset=offset, limit=limit)
    return {"ok": True, **data}


@router.get("/jobs/{job_id}/resources")
def job_resources(job_id: str) -> Dict[str, Any]:
    manager = get_manager()
    _guard(manager.get, job_id)
    return {"ok": True, "sample": manager.resources(job_id)}


# ---------------------------------------------------------------------------
# 过程图像（M2-06）
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str) -> Dict[str, Any]:
    manager = get_manager()
    data = _guard(manager.artifacts, job_id)
    for img in data.get("images", []):
        img["url"] = f"/api/train/jobs/{job_id}/image?name={img['name']}"
    for img in data.get("preview", []):
        img["url"] = f"/api/train/jobs/{job_id}/image?name={img['name']}"
    return {"ok": True, **data}


@router.get("/jobs/{job_id}/image")
def job_image(job_id: str, name: str = Query(..., description="训练目录下的图像文件名")):
    """返回训练目录内的过程图像。

    只接受相对文件名（不接受路径），并校验解析后仍在训练目录内，防止越权读取。
    """
    manager = get_manager()
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="name 必须是训练目录下的文件名")
    path = _guard(manager.artifact_path, job_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"图像不存在: {name}")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# WebSocket 实时推送（M2-04）
# ---------------------------------------------------------------------------


def make_event_bridge(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Dict[str, Any]]"):
    """把监控线程的同步回调转成事件循环里的队列投递。

    这段跨线程桥接是实时推送里唯一容易出错的地方，因此单独成函数便于测试。
    """

    def on_event(event: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    return on_event


@router.websocket("/jobs/{job_id}/ws")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    manager = get_manager()
    # 先完成握手再关闭，客户端才能拿到明确的关闭码 4404；
    # 若在 accept 之前 close，握手会被直接拒绝（客户端只看到 HTTP 403）。
    await websocket.accept()
    try:
        manager.get(job_id)
    except JobNotFound:
        await websocket.close(code=4404, reason="训练任务不存在")
        return

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    unsubscribe = manager.subscribe(job_id, make_event_bridge(loop, queue))

    try:
        await websocket.send_json(manager.snapshot(job_id))
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        # 客户端断开时 send 会抛任意异常，直接结束即可
        pass
    finally:
        unsubscribe()
