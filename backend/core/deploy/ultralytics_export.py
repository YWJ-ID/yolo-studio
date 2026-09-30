"""ultralytics 导出实现。

只在导出子进程里 import ultralytics。
逐个格式导出：某个格式失败不影响其它格式——用户往往只想先拿到能用的那个，
因此失败信息记在对应产物上，只要还有一个成功，整体就算部分成功。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from .result import Artifact, DeployResult, describe_artifact, now_iso
from .spec import DeploySpec


def export_kwargs(fmt: str, spec: DeploySpec) -> Dict[str, Any]:
    """组装 ultralytics export 的参数。纯函数，可单独测试。"""
    kwargs: Dict[str, Any] = {
        "format": fmt,
        "imgsz": spec.imgsz,
        "batch": spec.batch,
        "device": spec.device,
    }
    # 这些开关只对部分格式有意义，按格式给，避免 ultralytics 报未知参数
    if fmt == "onnx":
        kwargs["opset"] = spec.opset
        kwargs["simplify"] = spec.simplify
        kwargs["dynamic"] = spec.dynamic
        kwargs["half"] = spec.half
        kwargs["nms"] = spec.nms
    elif fmt == "torchscript":
        kwargs["half"] = spec.half
    elif fmt == "openvino":
        kwargs["half"] = spec.half
        kwargs["dynamic"] = spec.dynamic
    elif fmt == "engine":
        kwargs["half"] = spec.half
        kwargs["dynamic"] = spec.dynamic
        kwargs["simplify"] = spec.simplify
        kwargs["int8"] = spec.int8
    elif fmt in ("tflite", "coreml", "rknn"):
        kwargs["half"] = spec.half
        kwargs["int8"] = spec.int8
    kwargs.update(spec.extra)
    return kwargs


def run(spec: DeploySpec) -> int:
    """在导出子进程中逐个格式导出。返回退出码。"""
    from ultralytics import YOLO  # 延迟导入：只发生在导出子进程里

    started = time.time()
    result = DeployResult(
        weights=spec.weights,
        model_name=Path(spec.weights).name,
        task=spec.task,
        job_id=spec.job_id,
        deploy_id=spec.name,
        tag=spec.tag,
        imgsz=spec.imgsz,
        device=spec.device,
        requested=spec.normalized_formats(),
        created_at=now_iso(),
    )

    model = YOLO(spec.weights)
    copy_dir = spec.run_dir if spec.copy_artifacts else None

    for fmt in result.requested:
        t0 = time.time()
        try:
            out = model.export(**export_kwargs(fmt, spec))
            path = Path(str(out))
            if not path.exists():
                raise FileNotFoundError(f"导出返回的路径不存在: {path}")
            info = describe_artifact(path, copy_artifacts_dir=copy_dir)
            result.artifacts.append(
                Artifact(
                    format=fmt,
                    ok=True,
                    path=info["path"],
                    source_path=info["source_path"],
                    is_dir=info["is_dir"],
                    size=info["size"],
                    sha256=info["sha256"],
                    duration_sec=time.time() - t0,
                )
            )
        except Exception as exc:
            result.artifacts.append(
                Artifact(
                    format=fmt,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_sec=time.time() - t0,
                )
            )

    result.duration_sec = round(time.time() - started, 3)
    succeeded = result.succeeded()
    failed = result.failed()
    result.ok = bool(succeeded)
    if not succeeded:
        result.error = "所有格式导出失败：" + "；".join(
            f"{a.format}({a.error})" for a in failed
        )

    result.save(spec.result_path)
    return 0 if result.ok else 1


__all__ = ["export_kwargs", "run"]
