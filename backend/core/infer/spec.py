"""一次推理会话的参数契约。

与 `TrainSpec` / `EvalSpec` 同构：序列化后交给推理子进程，事后追溯
「这批结果是用什么权重、什么参数得到的」只看这一个文件。

注意这里是**会话级**参数（加载权重时确定），而 `conf` / `iou` 属于
**单帧级**参数，允许在推理过程中动态调整，因此单独放在 `InferOptions` 里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 任务类型
TASK_DETECT = "detect"
TASK_CLASSIFY = "classify"
TASK_SEGMENT = "segment"
TASKS = (TASK_DETECT, TASK_CLASSIFY, TASK_SEGMENT)

PROTOCOL_VERSION = 1


@dataclass
class InferSpec:
    """加载权重时的参数（会话级）。"""

    weights: str = ""
    # 显示用的模型名（默认取权重文件名）
    name: str = ""
    # 显式指定任务类型；空则由模型自行决定
    task: str = ""
    # 显式类别清单。优先于模型自带的类名；ONNX 等未写入类名时必须提供。
    classes: List[str] = field(default_factory=list)
    # 推理设备：cpu / 0 / 0,1 ... 空 = 自动
    device: str = "cpu"
    # 推理输入尺寸
    imgsz: int = 640
    # 是否半精度（GPU 上有意义）
    half: bool = False
    # 格式名；空则按路径推断（core.infer.formats.detect_weights_format）
    fmt: str = ""
    # 仅用于展示与追溯，不参与推理
    source: str = ""   # 权重来源：model_library / external / deploy
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------- 派生 ----------

    @property
    def display_name(self) -> str:
        return self.name or Path(self.weights).name

    def normalized_task(self) -> str:
        t = (self.task or "").strip().lower()
        if t and t not in TASKS:
            raise ValueError(f"未知任务类型: {self.task}，可选 {TASKS}")
        return t

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "name": self.name,
            "task": self.task,
            "classes": list(self.classes),
            "device": self.device,
            "imgsz": self.imgsz,
            "half": self.half,
            "fmt": self.fmt,
            "source": self.source,
            "note": self.note,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InferSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.weights:
            problems.append("weights 不能为空")
        else:
            p = Path(self.weights)
            if not p.exists():
                problems.append(f"权重不存在: {self.weights}")
        if self.imgsz <= 0:
            problems.append(f"imgsz 必须为正整数，收到 {self.imgsz}")
        try:
            self.normalized_task()
        except ValueError as exc:
            problems.append(str(exc))
        return problems


@dataclass
class InferOptions:
    """单帧推理参数（可随每帧调整）。"""

    conf: float = 0.25
    iou: float = 0.7
    # 最多保留多少个检测框（0 = 不限）
    max_det: int = 300
    # 只保留这些类别的下标；空 = 全部
    only_classes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "only_classes": list(self.only_classes),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "InferOptions":
        d = d or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not 0.0 <= self.conf <= 1.0:
            problems.append(f"conf 必须在 [0,1]，收到 {self.conf}")
        if not 0.0 <= self.iou <= 1.0:
            problems.append(f"iou 必须在 [0,1]，收到 {self.iou}")
        if self.max_det < 0:
            problems.append(f"max_det 不能为负，收到 {self.max_det}")
        return problems


__all__ = [
    "PROTOCOL_VERSION",
    "TASKS",
    "TASK_CLASSIFY",
    "TASK_DETECT",
    "TASK_SEGMENT",
    "InferOptions",
    "InferSpec",
]
