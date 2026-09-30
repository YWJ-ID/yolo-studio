"""训练 / 评估管理器的单例与二者的联动。

放在这里而不是各个路由文件里，是为了让「训练完成自动评估」有一个明确的归属地：
路由只做 HTTP 与参数校验，联动逻辑属于服务层。

core 层不认识评估与训练的关系，联动完全在此处组装。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.deploy import DeployManager
from core.eval import EvalManager, EvalSpec
from core.registry import ModelRegistry
from core.train import (
    STATUS_FINISHED,
    TrainingJob,
    TrainingManager,
    resolve_device,
)

from .config import settings

_lock = threading.RLock()
_training: Optional[TrainingManager] = None
_eval: Optional[EvalManager] = None
_registry: Optional[ModelRegistry] = None
_deploy: Optional[DeployManager] = None
_auto_eval_off = False


def _auto_eval_enabled() -> bool:
    """是否在训练完成后自动评估。用 YOLO_STUDIO_AUTO_EVAL=0 关闭。"""
    if _auto_eval_off:
        return False
    return os.environ.get("YOLO_STUDIO_AUTO_EVAL", "1").strip().lower() not in ("0", "false", "no")


def get_training_manager() -> TrainingManager:
    global _training
    if _training is None:
        with _lock:
            if _training is None:
                settings.ensure_dirs()
                manager = TrainingManager(
                    settings.runs_dir,
                    python=str(settings.python_executable),
                    device=resolve_device(settings.train_device),
                    weights_dir=str(settings.weights_dir),
                )
                manager.on_finish(_after_training_finish)
                _training = manager
    return _training


def get_eval_manager() -> EvalManager:
    global _eval
    if _eval is None:
        with _lock:
            if _eval is None:
                settings.ensure_dirs()
                manager = EvalManager(
                    settings.evals_dir,
                    python=str(settings.python_executable),
                    device=resolve_device(settings.train_device),
                )
                manager.on_finish(_after_eval_finish)
                _eval = manager
    return _eval


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                settings.ensure_dirs()
                _registry = ModelRegistry(settings.models_dir)
    return _registry


def get_deploy_manager() -> DeployManager:
    global _deploy
    if _deploy is None:
        with _lock:
            if _deploy is None:
                settings.ensure_dirs()
                manager = DeployManager(
                    settings.deploys_dir,
                    python=str(settings.python_executable),
                    device=resolve_device(settings.train_device),
                )
                manager.on_finish(_after_deploy_finish)
                _deploy = manager
    return _deploy


def set_managers(
    training: Optional[TrainingManager] = None,
    evaluation: Optional[EvalManager] = None,
    registry: Optional[ModelRegistry] = None,
    deploy: Optional[DeployManager] = None,
) -> None:
    """替换单例。供测试注入受控管理器。

    注入时顺带把联动回调挂上，保证「注入的管理器」和「自建的管理器」
    行为一致——否则很容易漏掉自动评估/自动注册/产物挂载。
    """
    global _training, _eval, _registry, _deploy
    with _lock:
        if training is not None:
            _training = training
        if evaluation is not None:
            _eval = evaluation
        if registry is not None:
            _registry = registry
        if deploy is not None:
            _deploy = deploy
    if training is not None:
        training.on_finish(_after_training_finish)
    if evaluation is not None:
        evaluation.on_finish(_after_eval_finish)
    if deploy is not None:
        deploy.on_finish(_after_deploy_finish)


def reset_managers() -> None:
    global _training, _eval, _registry, _deploy
    with _lock:
        _training = None
        _eval = None
        _registry = None
        _deploy = None


def set_auto_eval(enabled: bool) -> None:
    """打开/关闭训练完成后的自动评估。"""
    global _auto_eval_off
    _auto_eval_off = not enabled


def shutdown_managers() -> None:
    global _training, _eval, _deploy
    with _lock:
        training, evaluation, deploy = _training, _eval, _deploy
    if training is not None:
        training.shutdown()
    if evaluation is not None:
        evaluation.shutdown()
    if deploy is not None:
        deploy.shutdown()


# ---------------------------------------------------------------------------
# 训练完成 -> 自动评估
# ---------------------------------------------------------------------------


def _read_data_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def pick_split(data_yaml: str) -> str:
    """选评估划分：data.yaml 声明了 test 就用 test，否则用 val。

    用 test 才能得到「没参与过训练也不参与调参」的指标；没有 test 就退而求其次。
    """
    configured = os.environ.get("YOLO_STUDIO_AUTO_EVAL_SPLIT", "auto").strip().lower()
    if configured in ("val", "test", "train"):
        return configured
    data = _read_data_yaml(Path(data_yaml))
    if data.get("test"):
        return "test"
    return "val"


def model_for_job(job: TrainingJob) -> Tuple[Optional[Path], bool]:
    """取该训练任务可用于评估的权重：优先 best.pt，退而用 last.pt。

    返回 (路径, 是否 best)。
    """
    run_dir = Path(job.run_dir)
    best = run_dir / "weights" / "best.pt"
    if best.is_file():
        return best, True
    last = run_dir / "weights" / "last.pt"
    if last.is_file():
        return last, False
    return None, False


def auto_eval_spec(job: TrainingJob, evals_dir: Optional[Path] = None) -> Optional[EvalSpec]:
    """为一次已完成的训练构造自动评估参数；条件不满足时返回 None。"""
    spec = job.spec or {}
    data_yaml = str(spec.get("data_yaml") or "")
    if not data_yaml or not Path(data_yaml).is_file():
        return None
    weights, is_best = model_for_job(job)
    if weights is None:
        return None

    split = pick_split(data_yaml)
    return EvalSpec(
        weights=str(weights),
        data_yaml=data_yaml,
        project=str(evals_dir or settings.evals_dir),
        name="",  # 由 EvalManager 生成
        task=str(spec.get("task") or "detect"),
        split=split,
        imgsz=int(spec.get("imgsz") or 640),
        batch=int(spec.get("batch") or 16),
        device=resolve_device(settings.train_device),
        job_id=job.id,
        tag=f"{job.id}_{split}",
    )


def _after_training_finish(job: TrainingJob) -> None:
    """训练收尾回调：成功则注册模型并自动评估。任何问题都只记录，不抛给训练流程。"""
    if job.status != STATUS_FINISHED:
        return

    # 先注册模型（与是否自动评估无关）：模型库应当记录每一次成功的训练
    try:
        get_registry().register_from_training(job)
    except Exception as exc:
        try:
            job.message = f"{job.message}；模型注册失败: {exc}"
        except Exception:
            pass

    if not _auto_eval_enabled():
        return
    try:
        manager = get_eval_manager()
        spec = auto_eval_spec(job, evals_dir=manager.evals_dir)
        if spec is None:
            return
        manager.start(spec)
    except Exception as exc:  # 自动评估失败不能影响训练结果
        try:
            job.message = f"{job.message}；自动评估未启动: {exc}"
        except Exception:
            pass


def _after_eval_finish(job, result) -> None:
    """评估收尾回调：把结果挂到对应模型卡片上。"""
    if result is None or not job.spec:
        return
    model_id = job.spec.get("job_id") or ""
    if not model_id:
        return
    try:
        get_registry().attach_eval(model_id, job, result)
    except Exception:
        pass


def _after_deploy_finish(job, result) -> None:
    """导出收尾回调：把产物挂到对应模型卡片上。"""
    if result is None or not job.spec:
        return
    model_id = job.spec.get("job_id") or ""
    if not model_id:
        return
    try:
        get_registry().attach_deploy(model_id, job, result)
    except Exception:
        pass


def model_weights_for_deploy(model_id: str):
    """取某模型可用于导出的权重（优先 best.pt）。"""
    card = get_registry().get(model_id)
    if card is None:
        return None
    weights = (card.weights or {}).get("best") or (card.weights or {}).get("last")
    if not weights:
        return None
    return Path(weights)


__all__ = [
    "auto_eval_spec",
    "get_deploy_manager",
    "get_eval_manager",
    "get_registry",
    "get_training_manager",
    "model_for_job",
    "model_weights_for_deploy",
    "pick_split",
    "reset_managers",
    "set_auto_eval",
    "set_managers",
    "shutdown_managers",
]
