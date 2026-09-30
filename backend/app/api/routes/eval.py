"""评估接口（M3-01 / M3-02）。

职责：参数校验 -> 调用 core.eval -> 序列化 + 事件转发。
「训练完成自动评估」的联动在 `app/services.py`，本文件不掺和。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from core.eval import (
    EvalJobNotFound,
    EvalManager,
    EvalSpec,
    EvalStateError,
    SPLITS,
    write_eval_report,
)
from core.train import resolve_device

from ...config import settings
from ...schemas import (
    EvalJobResponse,
    EvalJobsResponse,
    EvalReportRequest,
    EvalReportResponse,
    EvalRequest,
    EvalResultResponse,
)
from ...services import get_eval_manager, get_training_manager, pick_split

router = APIRouter(prefix="/api/eval", tags=["eval"])


def get_manager() -> EvalManager:
    return get_eval_manager()


def set_manager(manager: Optional[EvalManager]) -> None:
    """替换单例。供测试注入受控管理器。"""
    from ...services import set_managers

    set_managers(evaluation=manager)


def _guard(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except EvalJobNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EvalStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_source(req: EvalRequest) -> Dict[str, Any]:
    """确定权重与 data.yaml：来自训练任务或显式指定。"""
    weights = req.weights
    data_yaml = req.data_yaml
    task = req.task
    imgsz = req.imgsz
    batch = req.batch
    job_id = req.job_id or ""

    if req.job_id:
        from ...services import model_for_job

        job = _guard(get_training_manager().get, req.job_id)
        job_spec = job.spec or {}
        if not weights:
            path, _is_best = model_for_job(job)
            if path is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"训练任务 {req.job_id} 没有可用权重（weights/best.pt 与 last.pt 都不存在）",
                )
            weights = str(path)
        if not data_yaml:
            data_yaml = str(job_spec.get("data_yaml") or "")
        if req.task == "detect" and job_spec.get("task"):
            task = str(job_spec["task"])
        if req.imgsz == 640 and job_spec.get("imgsz"):
            imgsz = int(job_spec["imgsz"])
        if req.batch == 16 and job_spec.get("batch"):
            batch = int(job_spec["batch"])

    if not weights or not data_yaml:
        raise HTTPException(
            status_code=400,
            detail="需要提供 job_id，或同时提供 weights 与 data_yaml",
        )
    if not Path(weights).is_file():
        raise HTTPException(status_code=400, detail=f"权重文件不存在: {weights}")
    if not Path(data_yaml).is_file():
        raise HTTPException(status_code=400, detail=f"data.yaml 不存在: {data_yaml}")

    return {
        "weights": weights,
        "data_yaml": data_yaml,
        "task": task,
        "imgsz": imgsz,
        "batch": batch,
        "job_id": job_id,
    }


@router.get("/splits")
def list_splits() -> Dict[str, Any]:
    return {"ok": True, "splits": list(SPLITS)}


@router.post("/jobs", response_model=EvalJobResponse)
def create_job(req: EvalRequest) -> EvalJobResponse:
    """新建评估任务。split 传 auto 时按 data.yaml 是否声明 test 自动选择。"""
    if req.split not in SPLITS and req.split != "auto":
        raise HTTPException(
            status_code=400, detail=f"未知 split: {req.split}，可选 {list(SPLITS)} 或 auto"
        )

    source = _resolve_source(req)
    split = req.split
    if split == "auto":
        split = pick_split(source["data_yaml"])

    spec = EvalSpec(
        weights=source["weights"],
        data_yaml=source["data_yaml"],
        project=str(get_manager().evals_dir),
        name="",
        task=source["task"],
        split=split,
        imgsz=source["imgsz"],
        batch=source["batch"],
        device=resolve_device(req.device or settings.train_device),
        workers=req.workers,
        conf=req.conf,
        iou=req.iou,
        job_id=source["job_id"],
        tag=req.tag or (f"{source['job_id']}_{split}" if source["job_id"] else split),
        extra=dict(req.extra),
    )
    job = _guard(get_manager().start, spec)
    return EvalJobResponse(job=job.to_dict())


@router.get("/jobs", response_model=EvalJobsResponse)
def list_jobs() -> EvalJobsResponse:
    return EvalJobsResponse(jobs=[j.to_dict() for j in get_manager().list()])


@router.get("/jobs/{eval_id}")
def get_job(eval_id: str) -> Dict[str, Any]:
    manager = get_manager()
    job = _guard(manager.get, eval_id)
    result = manager.result(eval_id)
    return {
        "ok": True,
        "job": job.to_dict(),
        "result": result.to_dict() if result else None,
        "logs": manager.logs(eval_id, limit=200),
        "resources": manager.resources(eval_id),
    }


@router.post("/jobs/{eval_id}/stop", response_model=EvalJobResponse)
def stop_job(eval_id: str) -> EvalJobResponse:
    job = _guard(get_manager().stop, eval_id)
    return EvalJobResponse(job=job.to_dict())


@router.get("/jobs/{eval_id}/result", response_model=EvalResultResponse)
def job_result(eval_id: str) -> EvalResultResponse:
    manager = get_manager()
    _guard(manager.get, eval_id)
    result = manager.result(eval_id)
    return EvalResultResponse(result=result.to_dict() if result else None)


@router.post("/jobs/{eval_id}/report", response_model=EvalReportResponse)
def job_report(eval_id: str, req: EvalReportRequest) -> EvalReportResponse:
    """生成自包含的单文件 HTML 评估报告，写在评估目录里（与产物同处，便于整体归档）。"""
    manager = get_manager()
    job = _guard(manager.get, eval_id)
    result = manager.result(eval_id)
    if result is None:
        raise HTTPException(status_code=409, detail="评估尚无结果，无法生成报告")

    slug = re.sub(r"[^0-9A-Za-z_\-]+", "_", req.out_name or eval_id)[:48].strip("_") or "eval_report"
    base = Path(job.run_dir).resolve()
    out_path = (base / f"{slug}.html").resolve()
    try:
        out_path.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法的输出名称") from None

    write_eval_report(
        result,
        out_path,
        title=req.title,
        images_dir=base,
        embed_images=req.embed_images,
    )
    return EvalReportResponse(
        path=str(out_path),
        url=f"/api/files/download?path={quote(str(out_path))}",
        size_bytes=out_path.stat().st_size,
    )


@router.get("/jobs/{eval_id}/logs")
def job_logs(
    eval_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    manager = get_manager()
    _guard(manager.get, eval_id)
    return {"ok": True, **manager.logs(eval_id, offset=offset, limit=limit)}


@router.get("/jobs/{eval_id}/artifacts")
def job_artifacts(eval_id: str) -> Dict[str, Any]:
    """评估过程图像：混淆矩阵、PR / F1 曲线、验证批次预测图。"""
    manager = get_manager()
    data = _guard(manager.artifacts, eval_id)
    for img in data.get("images", []):
        img["url"] = f"/api/eval/jobs/{eval_id}/image?name={img['name']}"
    return {"ok": True, **data}


@router.get("/jobs/{eval_id}/image")
def job_image(eval_id: str, name: str = Query(..., description="评估目录下的图像文件名")):
    manager = get_manager()
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="name 必须是评估目录下的文件名")
    path = _guard(manager.artifact_path, eval_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"图像不存在: {name}")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def make_event_bridge(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Dict[str, Any]]"):
    def on_event(event: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    return on_event


@router.websocket("/jobs/{eval_id}/ws")
async def job_ws(websocket: WebSocket, eval_id: str) -> None:
    manager = get_manager()
    await websocket.accept()
    try:
        manager.get(eval_id)
    except EvalJobNotFound:
        await websocket.close(code=4404, reason="评估任务不存在")
        return

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    unsubscribe = manager.subscribe(eval_id, make_event_bridge(loop, queue))
    try:
        await websocket.send_json(manager.snapshot(eval_id))
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        unsubscribe()
