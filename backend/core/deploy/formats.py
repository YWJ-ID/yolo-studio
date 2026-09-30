"""导出格式清单与可用性探测。

可用性用 `importlib.util.find_spec` 判断（不真正 import），避免为了回答
「能不能导出 ONNX」而把 onnx / openvino 这些重依赖加载进 API 进程。

清单里的 `requires` 是 ultralytics 真正需要的 Python 包名，
缺包时给出明确原因，而不是等用户在导出时报一个晦涩的 ImportError。
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 里程碑归属，便于前端分组展示
M4_PRIMARY = "M4-01"
M4_RKNN = "M4-02"


@dataclass(frozen=True)
class ExportFormat:
    name: str                 # ultralytics 的 format 取值
    label: str                # 中文说明
    suffix: str               # 产物扩展名（用于命名，目录型格式留空）
    requires: Tuple[str, ...] = ()
    needs_gpu: bool = False
    milestone: str = M4_PRIMARY
    note: str = ""


FORMATS: Tuple[ExportFormat, ...] = (
    ExportFormat(
        name="torchscript",
        label="TorchScript",
        suffix=".torchscript",
        requires=("torch",),
        note="PyTorch 自带，离线可用；适合在无 onnx 环境里先验证导出链路",
    ),
    ExportFormat(
        name="onnx",
        label="ONNX",
        suffix=".onnx",
        requires=("onnx",),
        note="越平台部署常用；开启 simplify 还需要 onnxslim",
    ),
    ExportFormat(
        name="openvino",
        label="OpenVINO",
        suffix="",  # 目录型产物
        requires=("openvino",),
        note="Intel CPU/GPU 推理，产物是目录",
    ),
    ExportFormat(
        name="engine",
        label="TensorRT Engine",
        suffix=".engine",
        requires=("tensorrt",),
        needs_gpu=True,
        note="需要 NVIDIA GPU 与 TensorRT，且 engine 与显卡/驱动绑定",
    ),
    ExportFormat(
        name="tflite",
        label="TensorFlow Lite",
        suffix=".tflite",
        requires=("tensorflow",),
        note="移动端/边缘常用",
    ),
    ExportFormat(
        name="coreml",
        label="CoreML",
        suffix=".mlpackage",
        requires=("coremltools",),
        note="Apple 生态，通常需要 macOS",
    ),
    ExportFormat(
        name="rknn",
        label="RKNN（瑞芯微 NPU）",
        suffix=".rknn",
        requires=("rknn",),
        milestone=M4_RKNN,
        note="需要 rknn-toolkit2，一般只在 Linux + 对应 Python 版本下可用；"
        "本机缺失时请按 rknn_model_zoo 的说明在独立环境里导出",
    ),
)

_BY_NAME: Dict[str, ExportFormat] = {f.name: f for f in FORMATS}

# 默认导出：ONNX（生态最通用）+ TorchScript（离线可验证）
DEFAULT_FORMATS: Tuple[str, ...] = ("onnx", "torchscript")


def get_format(name: str) -> ExportFormat:
    fmt = _BY_NAME.get(str(name))
    if fmt is None:
        raise ValueError(f"未知导出格式: {name}，可选 {sorted(_BY_NAME)}")
    return fmt


def is_known(name: str) -> bool:
    return str(name) in _BY_NAME


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def cuda_available() -> bool:
    """是否可用 CUDA。只在需要判断 engine 格式时调用。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def capability_report() -> List[Dict[str, object]]:
    """列出全部格式及其在本机的可用性。

    可用性 = 依赖齐全 且 （不需要 GPU 或 GPU 可用）。
    """
    out: List[Dict[str, object]] = []
    cuda = None
    for fmt in FORMATS:
        missing = [m for m in fmt.requires if not _installed(m)]
        reason = ""
        available = True
        if missing:
            available = False
            reason = "缺少 Python 包: " + ", ".join(missing)
        elif fmt.needs_gpu:
            if cuda is None:
                cuda = cuda_available()
            if not cuda:
                available = False
                reason = "需要 NVIDIA GPU / CUDA，本机不可用"
        out.append(
            {
                "name": fmt.name,
                "label": fmt.label,
                "suffix": fmt.suffix,
                "requires": list(fmt.requires),
                "needs_gpu": fmt.needs_gpu,
                "milestone": fmt.milestone,
                "note": fmt.note,
                "available": available,
                "reason": reason,
            }
        )
    return out


def unavailable(name: str) -> str:
    """返回该格式不可用的原因；可用时返回空串。"""
    for item in capability_report():
        if item["name"] == name:
            return str(item["reason"]) if not item["available"] else ""
    return f"未知导出格式: {name}"
