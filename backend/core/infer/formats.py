"""可推理的权重格式清单与可用性探测。

用 `importlib.util.find_spec` 判断依赖是否齐全（**不真正 import**），
避免为了回答「能不能用 ONNX 推理」而把 ort/openvino 拉进 API 进程。
与 `core/deploy/formats.py` 同风格，但侧重不同：

    deploy/formats.py  管「能不能**导出**成这个格式」
    infer/formats.py   管「能不能**用**这个格式推理」

两者不强行合并：同一格式的两个方向所需依赖本来就不同
（例如 ONNX 导出要 onnx，推理要 onnxruntime）。
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class InferFormat:
    name: str                 # 规范名（也是扩展名常用的标识）
    label: str                # 中文说明
    suffixes: Tuple[str, ...]  # 该格式对应的权重文件扩展名
    requires: Tuple[str, ...] = ()
    task_hint: str = ""       # 说明文字，写清这个格式的特点
    note: str = ""


FORMATS: Tuple[InferFormat, ...] = (
    InferFormat(
        name="pt",
        label="PyTorch (.pt)",
        suffixes=(".pt",),
        requires=("torch", "ultralytics"),
        task_hint="ultralytics 原生格式，检测 / 分类都能用；权重自带任务类型",
        note="本项目训练与评估产出的权重都是这个格式",
    ),
    InferFormat(
        name="onnx",
        label="ONNX (.onnx)",
        suffixes=(".onnx",),
        requires=("onnxruntime",),
        task_hint="跨平台推理常用。类名**不一定**有：ultralytics 导出时通常会把 names 写进 ONNX metadata，"
        "此时可自动读出；若没有，必须显式提供类别清单",
        note="由 M4 导出得到；推理走 onnxruntime 执行后端",
    ),
    InferFormat(
        name="torchscript",
        label="TorchScript (.torchscript)",
        suffixes=(".torchscript", ".pt"),
        requires=("torch",),
        task_hint="PyTorch 自带，无需额外运行时",
        note="由 M4 导出得到",
    ),
    InferFormat(
        name="openvino",
        label="OpenVINO",
        suffixes=("_openvino_model",),
        requires=("openvino",),
        task_hint="Intel CPU/GPU 推理",
        note="需要 openvino 运行时",
    ),
    InferFormat(
        name="engine",
        label="TensorRT Engine (.engine)",
        suffixes=(".engine",),
        requires=("tensorrt",),
        task_hint="需要 NVIDIA GPU，engine 与显卡/驱动绑定",
        note="本机无 CUDA 时不可用",
    ),
    InferFormat(
        name="tflite",
        label="TensorFlow Lite (.tflite)",
        suffixes=(".tflite",),
        requires=("tensorflow",),
        task_hint="移动端 / 边缘设备",
        note="",
    ),
    InferFormat(
        name="coreml",
        label="CoreML (.mlpackage)",
        suffixes=(".mlpackage",),
        requires=("coremltools",),
        task_hint="Apple 生态，通常需要 macOS",
        note="",
    ),
)

_BY_NAME: Dict[str, InferFormat] = {f.name: f for f in FORMATS}

# 扩展名 -> 格式名。注意 `.pt` 同时出现在 pt 与 torchscript 里：
# 单靠扩展名分不清是 ultralytics 权重还是 torchscript，需要看文件本身，
# 因此这里只做「候选提示」，最终格式由探测函数结合文件内容确定。
_SUFFIX_HINTS: Tuple[Tuple[str, str], ...] = (
    (".onnx", "onnx"),
    (".engine", "engine"),
    (".tflite", "tflite"),
    (".mlpackage", "coreml"),
    (".torchscript", "torchscript"),
)


def get_format(name: str) -> InferFormat:
    fmt = _BY_NAME.get(str(name))
    if fmt is None:
        raise ValueError(f"未知推理格式: {name}，可选 {sorted(_BY_NAME)}")
    return fmt


def is_known(name: str) -> bool:
    return str(name) in _BY_NAME


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_requirements(fmt: InferFormat) -> List[str]:
    return [m for m in fmt.requires if not _installed(m)]


def cuda_available() -> bool:
    """是否可用 CUDA。只在判断需要 GPU 的格式时调用。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def capability_report() -> List[Dict[str, object]]:
    """列出各推理格式在本机的可用性。

    可用性 = 依赖齐全 且 （不需要 GPU 的格式不需要额外判断）。
    不可用时给出**具体原因**（缺哪个包），不乐观假设。
    """
    out: List[Dict[str, object]] = []
    for fmt in FORMATS:
        missing = missing_requirements(fmt)
        available = not missing
        reason = ""
        if missing:
            reason = "缺少 Python 包: " + ", ".join(missing)
        out.append(
            {
                "name": fmt.name,
                "label": fmt.label,
                "suffixes": list(fmt.suffixes),
                "requires": list(fmt.requires),
                "task_hint": fmt.task_hint,
                "note": fmt.note,
                "available": available,
                "reason": reason,
            }
        )
    return out


def unavailable(name: str) -> str:
    """返回该格式不可用的原因；可用时返回空串。未知格式返回说明。"""
    if not is_known(name):
        return f"未知推理格式: {name}"
    fmt = get_format(name)
    missing = missing_requirements(fmt)
    if missing:
        return "缺少 Python 包: " + ", ".join(missing)
    return ""


def detect_weights_format(path: str) -> str:
    """由权重路径推断格式名。

    `.pt` 只按扩展名无法区分「ultralytics 权重」与「torchscript」，
    这里统一按 **pt** 处理；真正的任务与结构由加载时决定（见 ultralytics_infer）。
    目录型产物（OpenVINO）按其目录名后缀识别。
    """
    p = str(path).replace("\\", "/").rstrip("/")
    lower = p.lower()

    if lower.endswith("_openvino_model"):
        return "openvino"
    for suffix, name in _SUFFIX_HINTS:
        if lower.endswith(suffix):
            return name
    if lower.endswith(".pt"):
        return "pt"
    raise ValueError(
        f"无法从路径判断权重格式: {path}；"
        f"可用扩展名 {[s for f in FORMATS for s in f.suffixes]}"
    )


__all__ = [
    "FORMATS",
    "InferFormat",
    "capability_report",
    "cuda_available",
    "detect_weights_format",
    "get_format",
    "is_known",
    "missing_requirements",
    "unavailable",
]
