"""一次评估（val / test）的全部参数。

与 `TrainSpec` 同构：序列化到评估目录下的 `eval_spec.json`，
评估子进程读取它，事后追溯「这个指标是在哪份权重、哪个划分上算的」只看这一个文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# 可用于评估的划分
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

BACKEND_ULTRALYTICS = "ultralytics"

SPEC_FILE = "eval_spec.json"
RESULT_FILE = "eval_result.json"
LOG_FILE = "eval.log"


@dataclass
class EvalSpec:
    """评估参数。"""

    # 持久化数据可能来自旧版本或被手工编辑过，缺字段时用空串兜底，
    # 由 validate() 明确报错，而不是在加载阶段直接抛 TypeError 让整个任务列表加载失败。
    weights: str = ""
    data_yaml: str = ""
    project: str = ""
    name: str = ""
    backend: str = BACKEND_ULTRALYTICS
    task: str = "detect"
    split: str = SPLIT_VAL
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    workers: int = 0
    conf: float = 0.001
    iou: float = 0.6
    # 训练任务 id（若本次评估由某次训练触发，便于回查）
    job_id: str = ""
    # 展示用标签，如 "训练后自动评估" / "test 集复评"
    tag: str = ""
    python: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.project) / self.name

    @property
    def spec_path(self) -> Path:
        return self.run_dir / SPEC_FILE

    @property
    def result_path(self) -> Path:
        return self.run_dir / RESULT_FILE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "data_yaml": self.data_yaml,
            "project": self.project,
            "name": self.name,
            "backend": self.backend,
            "task": self.task,
            "split": self.split,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "workers": self.workers,
            "conf": self.conf,
            "iou": self.iou,
            "job_id": self.job_id,
            "tag": self.tag,
            "python": self.python,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.weights:
            problems.append("weights 不能为空")
        elif not Path(self.weights).is_file():
            problems.append(f"权重文件不存在: {self.weights}")
        if not self.data_yaml:
            problems.append("data_yaml 不能为空")
        elif not Path(self.data_yaml).is_file():
            problems.append(f"data.yaml 不存在: {self.data_yaml}")
        if not self.name:
            problems.append("name 不能为空")
        if not self.project:
            problems.append("project 不能为空")
        if self.split not in SPLITS:
            problems.append(f"未知 split: {self.split}，可选 {SPLITS}")
        if self.backend != BACKEND_ULTRALYTICS:
            problems.append(f"未知 backend: {self.backend}")
        return problems
