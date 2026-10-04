"""模型卡片：一个已训练模型的全部身份信息与血缘。

与数据集用 `dataset_card.json` 自描述同理：每个模型目录下放一份 `model_card.json`，
不需要数据库索引，也不会出现「库里有记录、磁盘上没文件」的错位。

一个训练任务对应一个模型目录（`<job_id>/`），评估结果以追加的方式挂到卡片上。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CARD_FILE = "model_card.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class ModelCard:
    """一个模型的注册信息。"""

    model_id: str
    name: str = ""
    task: str = "detect"
    created_at: str = ""
    updated_at: str = ""
    classes: List[str] = field(default_factory=list)
    # 权重文件（绝对路径）
    weights: Dict[str, str] = field(default_factory=dict)
    # 训练来源：job_id / run_dir / 关键超参 / 最优指标
    training: Dict[str, Any] = field(default_factory=dict)
    # 数据集血缘：data_yaml、数据集版本名、dataset_card 摘要
    dataset: Dict[str, Any] = field(default_factory=dict)
    # 评估结果索引（完整结果在各自评估目录的 eval_result.json）
    evals: List[Dict[str, Any]] = field(default_factory=list)
    # 导出产物索引（完整结果在各自导出目录的 deploy_result.json）
    deploys: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ---------- 查询 ----------

    def eval_for_split(self, split: str) -> Optional[Dict[str, Any]]:
        """取某个划分下最新的一次评估。同一划分重复评估时以最后一次为准。"""
        matches = [e for e in self.evals if e.get("split") == split]
        if not matches:
            return None
        return max(matches, key=lambda e: str(e.get("created_at", "")))

    def metric(self, key: str, split: Optional[str] = None) -> Optional[float]:
        """取指标：优先指定划分的评估结果，退而用训练过程的最优值。

        训练最优值的结构是 {best_epoch, best:{指标}}，因此要往里找一层。
        """
        ev = self.eval_for_split(split) if split else (self.evals[-1] if self.evals else None)
        if ev:
            value = (ev.get("overall") or {}).get(key)
            if value is not None:
                return value
        best = (self.training or {}).get("best") or {}
        nested = best.get("best") if isinstance(best.get("best"), dict) else {}
        value = nested.get(key, best.get(key))
        return value if isinstance(value, (int, float)) else None

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "task": self.task,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "classes": self.classes,
            "weights": self.weights,
            "training": self.training,
            "dataset": self.dataset,
            "evals": self.evals,
            "deploys": self.deploys,
            "notes": self.notes,
        }

    def summary(self) -> Dict[str, Any]:
        """列表页需要的精简信息。"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "task": self.task,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "dataset_name": (self.dataset or {}).get("name", ""),
            "job_id": (self.training or {}).get("job_id", ""),
            # training（本项目训练任务注册）| external（外部权重导入）
            "source": (self.training or {}).get("source", "training") or "training",
            "has_data_yaml": bool((self.dataset or {}).get("data_yaml")),
            "has_best": bool((self.weights or {}).get("best")),
            "best": (self.training or {}).get("best", {}),
            "eval_splits": sorted({e.get("split", "") for e in self.evals if e.get("split")}),
            "num_evals": len(self.evals),
            "deploy_formats": sorted(
                {a.get("format", "") for d in self.deploys for a in (d.get("artifacts") or []) if a.get("ok")}
            ),
            "num_deploys": len(self.deploys),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelCard":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    # ---------- 落盘 ----------

    def save(self, model_dir) -> Path:
        path = Path(model_dir) / CARD_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "ModelCard":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def touch(card: ModelCard) -> None:
    """更新时间戳。任何写操作前都要调用。"""
    card.updated_at = now_iso()
