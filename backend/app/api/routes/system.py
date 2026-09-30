"""系统信息与运行环境探测接口。"""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any, Dict, List

from fastapi import APIRouter

from core.deploy import capability_report

from ...config import settings
from ...schemas import EnvResponse, HealthResponse

router = APIRouter(prefix="/api/system", tags=["system"])

# 配置项 -> 对应的环境变量名。用于设置页展示「这个路径由哪个变量控制」。
_PATH_ENV = {
    "storage_dir": "YOLO_STUDIO_STORAGE",
    "datasets_dir": "YOLO_STUDIO_DATASETS",
    "runs_dir": "YOLO_STUDIO_RUNS",
    "weights_dir": "YOLO_STUDIO_WEIGHTS",
    "evals_dir": "YOLO_STUDIO_EVALS",
    "models_dir": "YOLO_STUDIO_MODELS",
    "deploys_dir": "YOLO_STUDIO_DEPLOYS",
    "uploads_dir": "YOLO_STUDIO_UPLOADS",
    "reports_dir": "YOLO_STUDIO_REPORTS",
    "db_path": "YOLO_STUDIO_DB",
}

# 可选依赖：缺了不影响基本功能，但对应能力会不可用（与能力探测口径一致）。
_OPTIONAL_DEPS = (
    ("torch", "TorchScript 导出 / CUDA 探测（经 ultralytics 安装）", "uv pip install torch"),
    ("psutil", "CPU / 内存资源监控", "uv pip install psutil"),
    ("pynvml", "NVIDIA GPU 利用率与显存监控", "uv pip install pynvml"),
    ("onnx", "ONNX 导出", "uv pip install onnx onnxslim"),
    ("onnxslim", "ONNX 图简化（simplify）", "uv pip install onnxslim"),
    ("onnxruntime", "ONNX 数值校验", "uv pip install onnxruntime"),
    ("openvino", "OpenVINO 导出", "uv pip install openvino"),
    ("tensorflow", "TensorFlow Lite 导出", "uv pip install tensorflow"),
    ("coremltools", "CoreML 导出（一般仅 macOS）", "uv pip install coremltools"),
    ("rknn", "RKNN 导出（需 rknn-toolkit2，一般仅 Linux）", "见 rknn_model_zoo 说明"),
    ("tensorrt", "TensorRT Engine 导出（需 NVIDIA GPU）", "uv pip install tensorrt"),
)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@router.get("/config")
def config() -> Dict[str, Any]:
    """只读的生效配置。

    这些值在后端**启动时**从环境变量读取，运行期不可修改；
    设置页据此展示「当前是什么、由哪个变量控制、改完需要重启」。
    """
    paths: List[Dict[str, Any]] = []
    for key, env_name in _PATH_ENV.items():
        value = getattr(settings, key)
        kind = "file" if key == "db_path" else "dir"
        paths.append(
            {
                "key": key,
                "env": env_name,
                "kind": kind,
                "value": str(value),
                "exists": value.is_file() if kind == "file" else value.is_dir(),
            }
        )

    return {
        "ok": True,
        "app": settings.app_name,
        "version": settings.version,
        "platform": platform.platform(),
        "api_port": settings.api_port,
        "python_executable": sys.executable,
        "training_python": str(settings.python_executable),
        "train_device": settings.train_device,
        "paths": paths,
        "allowed_roots": [str(p) for p in settings.allowed_roots],
        "files_read_unrestricted": not settings.allowed_roots,
        "frontend_dist": str(settings.frontend_dist),
        "frontend_dist_exists": settings.frontend_dist.is_dir(),
        "optional_dependencies": [
            {"name": name, "purpose": purpose, "install": install, "installed": _installed(name)}
            for name, purpose, install in _OPTIONAL_DEPS
        ],
        "export_formats": capability_report(),
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        storage_dir=str(settings.storage_dir),
    )


@router.get("/env", response_model=EnvResponse)
def env() -> EnvResponse:
    """探测训练环境：ultralytics / torch / CUDA / 训练解释器。

    前端据此决定是否显示 GPU 监控面板（无 CUDA 时降级为 CPU）。
    """
    info = EnvResponse(
        python=sys.version.split()[0],
        python_executable=sys.executable,
        training_python=str(settings.python_executable),
        platform=platform.platform(),
        train_device="cpu",
    )

    try:
        import ultralytics

        info.ultralytics = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        pass

    try:
        import torch

        info.torch = torch.__version__
        if torch.cuda.is_available():
            info.cuda_available = True
            info.cuda_device_count = torch.cuda.device_count()
            info.devices = [torch.cuda.get_device_name(i) for i in range(info.cuda_device_count)]
            info.train_device = "cuda:0"
        else:
            info.devices = ["cpu"]
            info.train_device = "cpu"
    except Exception:
        info.devices = ["cpu"]

    # 显式指定设备时以配置为准
    if settings.train_device and settings.train_device != "auto":
        info.train_device = settings.train_device

    return info
