"""模型库：注册、查询、把评估结果挂到模型卡片上。

存储方式：`<models_dir>/<job_id>/model_card.json`。
模型目录不复制权重文件，只记录权重**路径**——权重本来就躺在训练目录里，
复制一份会让「哪个才是当前模型」变得含糊。缺点是要保证训练目录不被移动，
因此卡片里同时记 run_dir，便于校验失效。

评估结果同样不复制：卡片只存 eval_id / split / overall 摘要，
完整逐类指标与混淆矩阵在评估目录的 eval_result.json 里，避免两处数据不一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from .model_card import CARD_FILE, ModelCard, now_iso, touch

# 数据集卡片文件名（由 core.export 写出）
DATASET_CARD = "dataset_card.json"


def _names_from_yaml(data_yaml: str) -> List[str]:
    """从 data.yaml 的 names 读类别名。names 可能是列表或 {索引: 名}。"""
    try:
        import yaml

        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    names = data.get("names")
    if isinstance(names, dict):
        try:
            return [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        except Exception:
            return [str(v) for v in names.values()]
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return []


def _dataset_info(data_yaml: str) -> Dict[str, Any]:
    """从 data.yaml 反查数据集血缘（同目录下的 dataset_card.json）。"""
    info: Dict[str, Any] = {"data_yaml": data_yaml}
    if not data_yaml:
        return info
    path = Path(data_yaml)
    card_path = path.parent / DATASET_CARD
    info["dataset_card"] = str(card_path) if card_path.is_file() else ""
    if card_path.is_file():
        try:
            import json

            card = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            card = {}
        if card:
            info["name"] = card.get("name", path.parent.name)
            info["task"] = card.get("task", "")
            info["classes"] = list(card.get("classes") or [])
            info["created_at"] = card.get("created_at", "")
            info["images_exported"] = card.get("images_exported", {})
            info["sources"] = card.get("sources", [])
            if not info["classes"]:
                info["classes"] = _names_from_yaml(data_yaml)
            return info

    # 没有 dataset_card（例如手工准备的数据集）：只能从 data.yaml 取类别
    info["name"] = path.parent.name
    info["classes"] = _names_from_yaml(data_yaml)
    return info


class ModelRegistry:
    """模型库，读写 `<models_dir>/<model_id>/model_card.json`。"""

    def __init__(self, models_dir) -> None:
        self.models_dir = Path(models_dir)

    # ---------- 路径 ----------

    def _dir(self, model_id: str) -> Path:
        base = self.models_dir.resolve()
        target = (base / model_id).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"非法的模型 id: {model_id}") from None
        if target == base:
            raise ValueError("模型 id 不能为空")
        return target

    def card_path(self, model_id: str) -> Path:
        return self._dir(model_id) / CARD_FILE

    # ---------- 查询 ----------

    def list(self) -> List[ModelCard]:
        if not self.models_dir.is_dir():
            return []
        cards: List[ModelCard] = []
        for card_file in sorted(self.models_dir.glob(f"*/{CARD_FILE}")):
            try:
                cards.append(ModelCard.load(card_file))
            except Exception:
                continue
        return sorted(cards, key=lambda c: (c.created_at, c.model_id), reverse=True)

    def get(self, model_id: str) -> Optional[ModelCard]:
        path = self.card_path(model_id)
        if not path.is_file():
            return None
        return ModelCard.load(path)

    # ---------- 写入 ----------

    def register_from_training(self, job) -> ModelCard:
        """根据一个已完成的训练任务注册模型。重复注册同一任务会更新原卡片。"""
        model_id = job.id
        spec = job.spec or {}
        data_yaml = str(spec.get("data_yaml") or "")
        run_dir = Path(job.run_dir)

        card = self.get(model_id)
        is_new = card is None
        if card is None:
            card = ModelCard(model_id=model_id, created_at=now_iso())

        card.name = card.name or model_id
        card.task = str(spec.get("task") or card.task or "detect")
        card.training = {
            "job_id": job.id,
            "run_dir": str(run_dir),
            "created_at": job.created_at,
            "ended_at": job.ended_at,
            "status": job.status,
            "epochs": spec.get("epochs"),
            "imgsz": spec.get("imgsz"),
            "batch": spec.get("batch"),
            "device": spec.get("device"),
            "seed": spec.get("seed"),
            "optimizer": spec.get("optimizer"),
            "weights_source": spec.get("weights"),
            "best": job.best or {},
            "attempts": job.attempts,
        }

        weights: Dict[str, str] = {}
        for key, name in (("best", "best.pt"), ("last", "last.pt")):
            candidate = run_dir / "weights" / name
            if candidate.is_file():
                weights[key] = str(candidate)
        card.weights = weights

        dataset = _dataset_info(data_yaml)
        if dataset:
            card.dataset = dataset
        if not card.classes:
            card.classes = list(dataset.get("classes") or [])

        if is_new:
            card.notes.append(f"由训练任务 {job.id} 注册")
        touch(card)
        card.save(self._dir(model_id))
        return card

    def attach_eval(self, model_id: str, eval_job, result) -> Optional[ModelCard]:
        """把一次评估结果挂到模型卡片上（同一 eval_id 重复挂载则覆盖）。"""
        card = self.get(model_id)
        if card is None:
            return None

        entry = {
            "eval_id": eval_job.id,
            "split": getattr(result, "split", "") or (eval_job.spec or {}).get("split", ""),
            "task": getattr(result, "task", ""),
            "created_at": getattr(result, "created_at", "") or now_iso(),
            "run_dir": eval_job.run_dir,
            "weights": getattr(result, "weights", ""),
            "overall": dict(getattr(result, "overall", {}) or {}),
            "num_classes": len(getattr(result, "per_class", []) or []),
            "artifacts": list(getattr(result, "artifacts", []) or []),
        }
        card.evals = [e for e in card.evals if e.get("eval_id") != eval_job.id]
        card.evals.append(entry)
        card.evals.sort(key=lambda e: str(e.get("created_at", "")))
        touch(card)
        card.save(self._dir(model_id))
        return card

    def link_training_job(self, model_id: str) -> Optional[str]:
        """返回该模型对应的训练任务 id（供接口用来触发重新评估）。"""
        card = self.get(model_id)
        if card is None:
            return None
        return (card.training or {}).get("job_id") or None

    def attach_deploy(self, model_id: str, deploy_job, result) -> Optional[ModelCard]:
        """把一次导出挂到模型卡片上（同一 deploy_id 重复挂载则覆盖）。"""
        card = self.get(model_id)
        if card is None:
            return None

        entry = {
            "deploy_id": deploy_job.id,
            "created_at": getattr(result, "created_at", "") or now_iso(),
            "run_dir": deploy_job.run_dir,
            "imgsz": getattr(result, "imgsz", None),
            "device": getattr(result, "device", ""),
            "ok": bool(getattr(result, "ok", False)),
            "artifacts": [
                a.to_dict() if hasattr(a, "to_dict") else dict(a)
                for a in (getattr(result, "artifacts", []) or [])
            ],
        }
        card.deploys = [d for d in card.deploys if d.get("deploy_id") != deploy_job.id]
        card.deploys.append(entry)
        card.deploys.sort(key=lambda d: str(d.get("created_at", "")))
        touch(card)
        card.save(self._dir(model_id))
        return card
