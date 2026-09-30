"""一次训练的全部参数。

`TrainSpec` 是 core 与子进程之间唯一的数据契约：
    * manager 把它写成 `run_dir/train_spec.json`；
    * 训练子进程读取它并调用具体后端；
    * 事后追溯「这次训练用了什么参数」只需读这个文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 任务类型
TASK_DETECT = "detect"
TASK_CLASSIFY = "classify"
TASKS = (TASK_DETECT, TASK_CLASSIFY)

# 后端名称
BACKEND_ULTRALYTICS = "ultralytics"

# 权重写法：以 .yaml 结尾表示从结构文件从零构建（不下载预训练权重）
YAML_SUFFIX = ".yaml"

# 未指定权重时按任务取默认值。两者都用 .yaml 结构文件：
# 从零训练不依赖网络，也不受「预训练权重是否已缓存」影响。
DEFAULT_WEIGHTS = {
    TASK_DETECT: "yolo11n.yaml",
    TASK_CLASSIFY: "yolo11-cls.yaml",
}


@dataclass
class TrainSpec:
    """训练参数。

    project + name 由调用方决定输出位置（等价于 ultralytics 的 project/name），
    `run_dir` 即两者拼接，是训练产物目录。
    """

    data_yaml: str = ""
    project: str = ""
    name: str = ""
    backend: str = BACKEND_ULTRALYTICS
    # 留空表示按 task 取 DEFAULT_WEIGHTS
    weights: str = ""
    task: str = TASK_DETECT
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    workers: int = 0
    seed: int = 42
    patience: int = 100
    optimizer: str = "auto"
    lr0: Optional[float] = None
    resume: bool = False
    # 训练使用的解释器；留空表示当前解释器
    python: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.project) / self.name

    @property
    def spec_path(self) -> Path:
        return self.run_dir / "train_spec.json"

    @property
    def is_from_scratch(self) -> bool:
        """权重是结构文件（从零训练）还是预训练权重。"""
        return str(self.weights).lower().endswith(YAML_SUFFIX)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_yaml": self.data_yaml,
            "project": self.project,
            "name": self.name,
            "backend": self.backend,
            "weights": self.weights,
            "task": self.task,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "workers": self.workers,
            "seed": self.seed,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "resume": self.resume,
            "python": self.python,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> List[str]:
        """返回参数问题列表，空列表表示合法。"""
        problems: List[str] = []
        if not self.data_yaml:
            problems.append("data_yaml 不能为空")
        elif not Path(self.data_yaml).is_file():
            problems.append(f"data.yaml 不存在: {self.data_yaml}")
        if not self.name:
            problems.append("name 不能为空")
        if not self.project:
            problems.append("project 不能为空")
        if self.task not in TASKS:
            problems.append(f"未知 task: {self.task}，可选 {TASKS}")
        if self.epochs <= 0:
            problems.append("epochs 必须为正")
        if self.imgsz <= 0:
            problems.append("imgsz 必须为正")
        if isinstance(self.batch, int) and self.batch <= 0:
            problems.append("batch 必须为正（或使用 0.5 这类比例值）")
        if self.backend not in (BACKEND_ULTRALYTICS,):
            problems.append(f"未知 backend: {self.backend}")
        return problems
