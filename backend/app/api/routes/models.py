"""模型库接口（M3-03 注册 / M3-04 对比）。

模型卡片只记录所有权重与评估的**引用**，本文件负责把它们读出来拼成
前端需要的视图；对比时按需加载各自的 eval_result.json。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from core.eval import SPLITS, compare_eval_results, load_result
from core.registry import ModelRegistry

from ...config import settings
from ...schemas import (
    ModelCompareRequest,
    ModelDetailResponse,
    ModelEvalRequest,
    ModelImportRequest,
    ModelListResponse,
    ModelRegisterRequest,
    ModelSummary,
)
from ...services import get_registry, get_training_manager, pick_split

router = APIRouter(prefix="/api/models", tags=["models"])


def get_registry_singleton() -> ModelRegistry:
    return get_registry()


def set_registry(registry: Optional[ModelRegistry]) -> None:
    """替换单例。供测试注入受控模型库。"""
    from ...services import set_managers

    set_managers(registry=registry)


def _require(model_id: str):
    card = get_registry_singleton().get(model_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    return card


# ---------------------------------------------------------------------------
# 注册与查询
# ---------------------------------------------------------------------------


@router.post("/register", response_model=ModelDetailResponse)
def register_model(req: ModelRegisterRequest) -> ModelDetailResponse:
    """把一次已完成的训练任务注册为模型（幂等：重复注册会更新卡片）。"""
    try:
        job = get_training_manager().get(req.job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"训练任务不存在: {req.job_id}") from exc

    if not (Path(job.run_dir) / "weights").is_dir():
        raise HTTPException(status_code=400, detail=f"训练任务 {req.job_id} 没有权重目录，无法注册")

    card = get_registry_singleton().register_from_training(job)
    return ModelDetailResponse(model=card.to_dict())


@router.post("/import", response_model=ModelDetailResponse)
def import_model(req: ModelImportRequest) -> ModelDetailResponse:
    """导入一个外部 `.pt` 权重进模型库（非训练任务）。

    导入后即可对它发起评估、对比、导出。评估需要 `data.yaml`（真值），
    未提供时卡片仍可导入，但评估会被拒。
    """
    try:
        card = get_registry_singleton().register_external(
            weights=req.weights,
            data_yaml=req.data_yaml,
            name=req.name,
            task=req.task,
            classes=req.classes,
            imgsz=req.imgsz,
            batch=req.batch,
            model_id=req.model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelDetailResponse(model=card.to_dict())


@router.get("", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    cards = get_registry_singleton().list()
    return ModelListResponse(models=[ModelSummary(**c.summary()) for c in cards])


@router.post("/compare")
def compare_models(req: ModelCompareRequest) -> Dict[str, Any]:
    """多模型对比（M3-04）。

    只比较同一划分下的评估结果；不同划分的数字不具备可比性，会在 notes 里提示。
    """
    if len(req.model_ids) < 1:
        raise HTTPException(status_code=400, detail="至少需要一个模型 id")

    registry = get_registry_singleton()
    items = []
    skipped: List[str] = []
    for model_id in req.model_ids:
        card = registry.get(model_id)
        if card is None:
            skipped.append(f"{model_id}(不存在)")
            continue
        entry = card.eval_for_split(req.split) if req.split else (card.evals[-1] if card.evals else None)
        if entry is None:
            skipped.append(f"{model_id}(没有{'该划分' if req.split else ''}评估结果)")
            continue
        result = load_result(entry.get("run_dir") or "")
        if result is None:
            skipped.append(f"{model_id}(结果文件缺失)")
            continue
        items.append((card.name or card.model_id, result))

    data = compare_eval_results(items)
    if skipped:
        data["notes"] = list(data.get("notes", [])) + [
            "已跳过：" + "、".join(skipped)
        ]
    return {"ok": True, **data}


@router.get("/{model_id}", response_model=ModelDetailResponse)
def get_model(model_id: str) -> ModelDetailResponse:
    card = _require(model_id)
    return ModelDetailResponse(model=card.to_dict())


@router.get("/{model_id}/result")
def get_model_result(model_id: str, split: Optional[str] = None) -> Dict[str, Any]:
    """取该模型某划分下最新一次评估的完整结果。"""
    card = _require(model_id)
    entry = card.eval_for_split(split) if split else (card.evals[-1] if card.evals else None)
    if entry is None:
        raise HTTPException(status_code=404, detail="该模型还没有评估结果")
    result = load_result(entry.get("run_dir") or "")
    if result is None:
        raise HTTPException(status_code=404, detail="评估结果文件缺失")
    return {"ok": True, "result": result.to_dict()}


# ---------------------------------------------------------------------------
# 用模型发起评估
# ---------------------------------------------------------------------------


@router.post("/{model_id}/eval")
def eval_model(model_id: str, req: ModelEvalRequest) -> Dict[str, Any]:
    """对已注册模型发起一次评估（复用 /api/eval 的调度）。"""
    from core.eval import EvalSpec
    from core.train import resolve_device

    from ...services import get_eval_manager

    card = _require(model_id)
    weights = (card.weights or {}).get("best") or (card.weights or {}).get("last")
    if not weights or not Path(weights).is_file():
        raise HTTPException(status_code=400, detail=f"模型 {model_id} 没有可用权重文件")

    data_yaml = str((card.dataset or {}).get("data_yaml") or "")
    if not data_yaml or not Path(data_yaml).is_file():
        raise HTTPException(status_code=400, detail=f"模型 {model_id} 关联的 data.yaml 不存在: {data_yaml}")

    if req.split not in SPLITS and req.split != "auto":
        raise HTTPException(status_code=400, detail=f"未知 split: {req.split}，可选 {list(SPLITS)} 或 auto")
    split = pick_split(data_yaml) if req.split == "auto" else req.split

    training = card.training or {}
    spec = EvalSpec(
        weights=str(weights),
        data_yaml=data_yaml,
        project=str(get_eval_manager().evals_dir),
        name="",
        task=card.task or "detect",
        split=split,
        imgsz=int(req.imgsz or training.get("imgsz") or 640),
        batch=int(req.batch or training.get("batch") or 16),
        device=resolve_device(req.device or settings.train_device),
        conf=req.conf,
        iou=req.iou,
        job_id=model_id,
        tag=req.tag or f"{model_id}_{split}",
    )
    job = get_eval_manager().start(spec)
    return {"ok": True, "job": job.to_dict()}
