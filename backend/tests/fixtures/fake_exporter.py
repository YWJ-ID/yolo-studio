"""测试用假导出进程。

模拟 ultralytics export 的外部行为：打印日志、按格式产出产物文件、写出 deploy_result.json。
用环境变量控制失败与耗时，便于验证部分成功、全部失败、停止等路径。

环境变量（全部可选）：
    FAKE_DEPLOY_DELAY  耗时秒（默认 0.2）
    FAKE_DEPLOY_FAIL   逗号分隔的格式名，这些格式导出失败（默认无）
    FAKE_DEPLOY_ALLFAIL=1 时所有格式都失败
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

FORMAT_SUFFIX = {
    "torchscript": ".torchscript",
    "onnx": ".onnx",
    "openvino": "",       # 目录型
    "engine": ".engine",
    "tflite": ".tflite",
    "coreml": ".mlpackage",
    "rknn": ".rknn",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def main() -> int:
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    run_dir = spec_path.parent

    delay = float(os.environ.get("FAKE_DEPLOY_DELAY", "0.2"))
    fail = {x for x in os.environ.get("FAKE_DEPLOY_FAIL", "").split(",") if x}
    all_fail = os.environ.get("FAKE_DEPLOY_ALLFAIL", "0") == "1"

    formats = spec.get("formats") or ["onnx"]
    stem = Path(str(spec.get("weights", "model.pt"))).stem
    copy_dir = run_dir if spec.get("copy_artifacts", True) else Path(spec.get("weights", ".")).parent

    print(f"fake exporter start formats={formats}", flush=True)

    artifacts = []
    for fmt in formats:
        t0 = time.time()
        time.sleep(delay)
        if all_fail or fmt in fail:
            print(f"  {fmt}: failed", flush=True)
            artifacts.append({
                "format": fmt, "ok": False, "path": "", "source_path": "", "is_dir": False,
                "size": 0, "sha256": "", "duration_sec": round(time.time() - t0, 3),
                "error": f"RuntimeError: fake export failed for {fmt}", "name": "",
            })
            continue

        suffix = FORMAT_SUFFIX.get(fmt, ".bin")
        if suffix:
            # 产物写在权重旁边（模仿 ultralytics），需要时再复制到导出目录
            source = Path(spec.get("weights", ".")).parent / f"{stem}{suffix}"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"fake-" + fmt.encode("utf-8") + b"-payload")
            target = source
            is_dir = False
            if spec.get("copy_artifacts", True):
                target_dir = run_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / source.name
                target.write_bytes(source.read_bytes())
        else:
            source = Path(spec.get("weights", ".")).parent / f"{stem}_{fmt}_model"
            source.mkdir(parents=True, exist_ok=True)
            (source / "model.xml").write_bytes(b"fake-openvino")
            target = source
            is_dir = True

        artifacts.append({
            "format": fmt, "ok": True, "path": str(target), "source_path": str(source),
            "is_dir": is_dir, "size": _size(target),
            "sha256": "" if is_dir else _sha256(target),
            "duration_sec": round(time.time() - t0, 3), "error": "",
            "name": target.name,
        })
        print(f"  {fmt}: ok -> {target}", flush=True)

    ok = any(a["ok"] for a in artifacts)
    result = {
        "ok": ok,
        "error": "" if ok else "所有格式导出失败：" + "；".join(
            f"{a['format']}({a['error']})" for a in artifacts if not a["ok"]
        ),
        "weights": spec.get("weights", ""),
        "model_name": Path(str(spec.get("weights", ""))).name,
        "task": spec.get("task", "detect"),
        "job_id": spec.get("job_id", ""),
        "deploy_id": spec.get("name", ""),
        "tag": spec.get("tag", ""),
        "imgsz": spec.get("imgsz", 640),
        "device": spec.get("device", "cpu"),
        "requested": formats,
        "artifacts": artifacts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(delay * len(formats), 3),
    }
    (run_dir / "deploy_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("fake exporter done", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
