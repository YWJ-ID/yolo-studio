"""部署导出接口（M4-01 / M4-02 / M4-03）。

职责：参数校验 -> 调用 core.deploy -> 序列化 + 事件转发。
产物挂到模型库的联动在 `app/services.py`。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from core.deploy import (
    DEFAULT_FORMATS,
    DeployJobNotFound,
    DeployManager,
    DeploySpec,
    DeployStateError,
    capability_report,
)
from core.train import resolve_device

from ...config import settings
from ...schemas import (
    DeployFormatInfo,
    DeployFormatsResponse,
    DeployJobResponse,
    DeployJobsResponse,
    DeployRequest,
    DeployResultResponse,
    DeployVerifyResponse,
)
from ...services import get_deploy_manager, get_registry, get_training_manager

router = APIRouter(prefix="/api/deploy", tags=["deploy"])


def get_manager() -> DeployManager:
    return get_deploy_manager()


def set_manager(manager: Optional[DeployManager]) -> None:
    """替换单例。供测试注入受控管理器。"""
    from ...services import set_managers

    set_managers(deploy=manager)


def _guard(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except DeployJobNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DeployStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/formats", response_model=DeployFormatsResponse)
def list_formats() -> DeployFormatsResponse:
    """列出全部导出格式及本机可用性（缺哪个包会写在 reason 里）。"""
    return DeployFormatsResponse(
        formats=[DeployFormatInfo(**f) for f in capability_report()],
        defaults=list(DEFAULT_FORMATS),
    )


def _resolve_weights(req: DeployRequest) -> Dict[str, Any]:
    """确定权重与任务类型：优先显式 weights，其次模型库，最后训练任务。"""
    weights = req.weights
    task = req.task
    job_id = ""

    if not weights and req.model_id:
        card = get_registry().get(req.model_id)
        if card is None:
            raise HTTPException(status_code=404, detail=f"模型不存在: {req.model_id}")
        weights = (card.weights or {}).get("best") or (card.weights or {}).get("last")
        task = card.task or task
        job_id = req.model_id
        if not weights:
            raise HTTPException(status_code=400, detail=f"模型 {req.model_id} 没有可用权重")

    if not weights and req.job_id:
        job = _guard(get_training_manager().get, req.job_id)
        from ...services import model_for_job

        path, _is_best = model_for_job(job)
        if path is None:
            raise HTTPException(status_code=400, detail=f"训练任务 {req.job_id} 没有可用权重")
        weights = str(path)
        task = str((job.spec or {}).get("task") or task)
        job_id = req.job_id

    if not weights:
        raise HTTPException(status_code=400, detail="需要提供 model_id、job_id 或 weights")
    if not Path(weights).is_file():
        raise HTTPException(status_code=400, detail=f"权重文件不存在: {weights}")

    return {"weights": str(weights), "task": task, "job_id": job_id}


@router.post("/jobs", response_model=DeployJobResponse)
def create_job(req: DeployRequest) -> DeployJobResponse:
    source = _resolve_weights(req)
    spec = DeploySpec(
        weights=source["weights"],
        project=str(get_manager().deploys_dir),
        name="",
        job_id=source["job_id"],
        task=source["task"],
        formats=list(req.formats),
        imgsz=req.imgsz,
        batch=req.batch,
        device=resolve_device(req.device or settings.train_device),
        half=req.half,
        dynamic=req.dynamic,
        simplify=req.simplify,
        opset=req.opset,
        int8=req.int8,
        nms=req.nms,
        copy_artifacts=req.copy_artifacts,
        tag=req.tag,
    )
    job = _guard(get_manager().start, spec)
    return DeployJobResponse(job=job.to_dict())


@router.get("/jobs", response_model=DeployJobsResponse)
def list_jobs() -> DeployJobsResponse:
    return DeployJobsResponse(jobs=[j.to_dict() for j in get_manager().list()])


@router.get("/jobs/{deploy_id}")
def get_job(deploy_id: str) -> Dict[str, Any]:
    manager = get_manager()
    job = _guard(manager.get, deploy_id)
    result = manager.result(deploy_id)
    return {
        "ok": True,
        "job": job.to_dict(),
        "result": result.to_dict() if result else None,
        "logs": manager.logs(deploy_id, limit=200),
    }


@router.post("/jobs/{deploy_id}/stop", response_model=DeployJobResponse)
def stop_job(deploy_id: str) -> DeployJobResponse:
    job = _guard(get_manager().stop, deploy_id)
    return DeployJobResponse(job=job.to_dict())


@router.get("/jobs/{deploy_id}/result", response_model=DeployResultResponse)
def job_result(deploy_id: str) -> DeployResultResponse:
    manager = get_manager()
    _guard(manager.get, deploy_id)
    result = manager.result(deploy_id)
    return DeployResultResponse(result=result.to_dict() if result else None)


@router.get("/jobs/{deploy_id}/verify", response_model=DeployVerifyResponse)
def verify_job(deploy_id: str) -> DeployVerifyResponse:
    """校验产物完整性：重新计算大小与 sha256，和导出时的基线比对（M4-03）。"""
    manager = get_manager()
    _guard(manager.get, deploy_id)
    report = _guard(manager.verify, deploy_id)
    return DeployVerifyResponse(verify=report.to_dict())


@router.get("/jobs/{deploy_id}/logs")
def job_logs(
    deploy_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    manager = get_manager()
    _guard(manager.get, deploy_id)
    return {"ok": True, **manager.logs(deploy_id, offset=offset, limit=limit)}


@router.get("/jobs/{deploy_id}/download")
def download_artifact(
    deploy_id: str,
    fmt: str = Query(..., description="格式名，如 onnx / torchscript"),
):
    """下载某个格式的产物。只允许下载本次导出目录内的文件。"""
    manager = get_manager()
    _guard(manager.get, deploy_id)
    result = manager.result(deploy_id)
    if result is None:
        raise HTTPException(status_code=409, detail="导出尚无结果")

    artifact = next((a for a in result.artifacts if a.format == fmt and a.ok), None)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"没有可用的 {fmt} 产物")
    path = Path(artifact.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"产物文件不存在: {artifact.path}")
    return FileResponse(path, filename=path.name)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def make_event_bridge(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Dict[str, Any]]"):
    def on_event(event: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    return on_event


@router.websocket("/jobs/{deploy_id}/ws")
async def job_ws(websocket: WebSocket, deploy_id: str) -> None:
    manager = get_manager()
    await websocket.accept()
    try:
        manager.get(deploy_id)
    except DeployJobNotFound:
        await websocket.close(code=4404, reason="导出任务不存在")
        return

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    unsubscribe = manager.subscribe(deploy_id, make_event_bridge(loop, queue))
    try:
        await websocket.send_json(manager.snapshot(deploy_id))
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        unsubscribe()
