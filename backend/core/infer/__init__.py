"""推理模块：对已训练权重做单帧 / 流式推理，用于实时验证与（后续的）批量预标注。

与训练/评估/导出的隔离策略一致：真正的推理跑在**常驻子进程**里，
API 进程不加载 torch/ultralytics。

与它们不同的是生命周期：训练/评估/导出是一次性任务，本模块要**常驻**
（起进程要几秒，逐帧起不可用），因此用 stdin/stdout 的 JSON Lines 协议通信。
"""

from __future__ import annotations

from .formats import (
    FORMATS,
    InferFormat,
    capability_report,
    cuda_available,
    detect_weights_format,
    get_format,
    is_known,
    unavailable,
)
from .result import Detection, FrameResult, build_frame_result
from .session import (
    DEFAULT_FRAME_TIMEOUT,
    DEFAULT_LOAD_TIMEOUT,
    InferError,
    InferSession,
)
from .spec import (
    PROTOCOL_VERSION,
    TASKS,
    TASK_CLASSIFY,
    TASK_DETECT,
    TASK_SEGMENT,
    InferOptions,
    InferSpec,
)

__all__ = [
    "DEFAULT_FRAME_TIMEOUT",
    "DEFAULT_LOAD_TIMEOUT",
    "Detection",
    "FORMATS",
    "FrameResult",
    "InferError",
    "InferFormat",
    "InferOptions",
    "InferSession",
    "InferSpec",
    "PROTOCOL_VERSION",
    "TASKS",
    "TASK_CLASSIFY",
    "TASK_DETECT",
    "TASK_SEGMENT",
    "build_frame_result",
    "capability_report",
    "cuda_available",
    "detect_weights_format",
    "get_format",
    "is_known",
    "unavailable",
]
