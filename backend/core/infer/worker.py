"""常驻推理子进程。

    python -m core.infer.worker

与训练/评估/导出不同，这里**不是一次性子进程**：起一个进程要几秒
（import torch），逐帧起进程不可用，因此进程常驻，
通过 stdin/stdout 的 **JSON Lines** 协议与主进程通信。

协议（每行一个 JSON）：

  主进程 -> 子进程
    {"type":"load",  "spec": {...InferSpec...}}
        -> {"type":"loaded","ok":true,"task":..,"classes":[..],"num_classes":..}
        -> {"type":"loaded","ok":false,"error":".."}

    {"type":"frame", "id":7, "jpeg":"<base64>", "options":{..InferOptions..}}
        -> {"type":"result","id":7,"ok":true,"task":..,"detections":[..],...}
        -> {"type":"result","id":7,"ok":false,"error":".."}

    {"type":"close"}
        -> {"type":"bye"} 然后退出

用 base64 传 JPEG 而不是二进制帧：协议简单、便于人工排查。
一帧几十 KB，本机 / 局域网完全够用。

退出码：0 = 正常关闭，1 = 致命错误，2 = 参数错误。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback
from typing import Any, Dict, Optional

from .spec import InferOptions, InferSpec


def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _decode_jpeg(b64: str):
    """base64 JPEG -> numpy BGR 数组。失败返回 None。"""
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1]  # -> BGR，ultralytics 默认按 BGR 处理 ndarray


class Worker:
    """持有已加载模型，逐帧推理。"""

    def __init__(self) -> None:
        self.model = None
        self.spec: Optional[InferSpec] = None
        self.names: Dict[int, str] = {}

    def handle_load(self, spec_dict: Dict[str, Any]) -> Dict[str, Any]:
        spec = InferSpec.from_dict(spec_dict)
        problems = spec.validate()
        if problems:
            return {"type": "loaded", "ok": False, "error": "; ".join(problems)}

        from . import ultralytics_infer as impl

        try:
            model = impl.load_model(spec)
            info = impl.describe_model(model, spec)
        except Exception as exc:
            return {
                "type": "loaded",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        self.model = model
        self.spec = spec
        self.names = {i: n for i, n in enumerate(info["classes"])}
        return {"type": "loaded", "ok": True, **info}

    def handle_frame(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        frame_id = msg.get("id")
        if self.model is None or self.spec is None:
            return {"type": "result", "id": frame_id, "ok": False,
                    "error": "模型尚未加载"}

        options = InferOptions.from_dict(msg.get("options"))
        problems = options.validate()
        if problems:
            return {"type": "result", "id": frame_id, "ok": False,
                    "error": "; ".join(problems)}

        try:
            source = _decode_jpeg(msg["jpeg"]) if msg.get("jpeg") else msg.get("path")
        except Exception as exc:
            return {"type": "result", "id": frame_id, "ok": False,
                    "error": f"图像解码失败: {type(exc).__name__}: {exc}"}

        if source is None:
            return {"type": "result", "id": frame_id, "ok": False,
                    "error": "缺少 jpeg 或 path"}

        import time

        from . import ultralytics_infer as impl

        started = time.perf_counter()
        try:
            payload = impl.predict_image(self.model, source, self.spec, options)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return {"type": "result", "id": frame_id, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"type": "result", "id": frame_id, "duration_ms": duration_ms, **payload}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="core.infer.worker",
                                     description="YOLO Studio 常驻推理子进程")
    parser.add_argument("--model", help="启动时预加载的权重路径（可选）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    worker = Worker()

    if args.model:
        early = worker.handle_load(
            InferSpec(weights=args.model, imgsz=args.imgsz, device=args.device).to_dict()
        )
        _emit(early)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"type": "error", "error": f"非法 JSON: {exc}"})
            continue

        mtype = msg.get("type")
        if mtype == "load":
            _emit(worker.handle_load(msg.get("spec") or {}))
        elif mtype == "frame":
            _emit(worker.handle_frame(msg))
        elif mtype == "close":
            _emit({"type": "bye"})
            return 0
        elif mtype == "ping":
            _emit({"type": "pong"})
        else:
            _emit({"type": "error", "error": f"未知指令: {mtype}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
