"""一次模型导出的参数。

与 TrainSpec / EvalSpec 同构：写成 `deploy_spec.json` 交给导出子进程，
事后追溯「这个 onnx 是怎么导出来的」只看这一个文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .formats import DEFAULT_FORMATS, get_format, is_known

SPEC_FILE = "deploy_spec.json"
RESULT_FILE = "deploy_result.json"
LOG_FILE = "deploy.log"


@dataclass
class DeploySpec:
    """导出参数。"""

    weights: str = ""
    project: str = ""
    name: str = ""
    # 任务 id（若由模型库触发，便于回查）
    job_id: str = ""
    task: str = "detect"
    formats: List[str] = field(default_factory=lambda: list(DEFAULT_FORMATS))
    imgsz: int = 640
    batch: int = 1
    device: str = "cpu"
    half: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: int = 0          # 0 = 用 ultralytics 默认
    int8: bool = False
    nms: bool = False
    # 把产物复制到导出目录（否则留在权重旁边，训练目录被清理就找不到）
    copy_artifacts: bool = True
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
            "project": self.project,
            "name": self.name,
            "job_id": self.job_id,
            "task": self.task,
            "formats": list(self.formats),
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "half": self.half,
            "dynamic": self.dynamic,
            "simplify": self.simplify,
            "opset": self.opset,
            "int8": self.int8,
            "nms": self.nms,
            "copy_artifacts": self.copy_artifacts,
            "tag": self.tag,
            "python": self.python,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeploySpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.weights:
            problems.append("weights 不能为空")
        elif not Path(self.weights).is_file():
            problems.append(f"权重文件不存在: {self.weights}")
        if not self.project:
            problems.append("project 不能为空")
        if not self.name:
            problems.append("name 不能为空")
        if not self.formats:
            problems.append("至少要选择一个导出格式")
        else:
            for name in self.formats:
                if not is_known(name):
                    problems.append(f"未知导出格式: {name}")
        if self.imgsz <= 0:
            problems.append("imgsz 必须为正")
        if self.batch <= 0:
            problems.append("batch 必须为正")
        if self.half and self.int8:
            problems.append("half 与 int8 不能同时开启")
        return problems

    def normalized_formats(self) -> List[str]:
        """去重并保持用户给定顺序，同时校验取值。"""
        seen: Dict[str, None] = {}
        for raw in self.formats:
            name = get_format(str(raw)).name
            seen.setdefault(name, None)
        return list(seen)
