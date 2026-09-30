"""测试用假评估进程。

模拟 ultralytics val 的外部行为：打印日志、产出过程图像、写出 eval_result.json。
指标由环境变量控制，便于构造"不同模型"以验证对比功能。

环境变量（全部可选）：
    FAKE_EVAL_DELAY   耗时秒（默认 0.2）
    FAKE_EVAL_EXIT    退出码（默认 0）
    FAKE_EVAL_MAP50   覆盖 mAP50（默认由 FAKE_EVAL_SEED 推导）
    FAKE_EVAL_SEED    指标基数，用于让不同模型产生不同指标
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ARTIFACTS = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
)


def main() -> int:
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    run_dir = spec_path.parent

    delay = float(os.environ.get("FAKE_EVAL_DELAY", "0.2"))
    exit_code = int(os.environ.get("FAKE_EVAL_EXIT", "0"))
    seed = float(os.environ.get("FAKE_EVAL_SEED", "1.0"))

    print(f"fake evaluator start split={spec.get('split')}", flush=True)
    time.sleep(delay)
    print("fake evaluator validating...", flush=True)

    names = {0: "square", 1: "circle"}
    base = min(0.95, 0.20 * seed)
    overall = {
        "precision": round(base + 0.10, 4),
        "recall": round(base + 0.05, 4),
        "f1": round(base + 0.07, 4),
        "mAP50": round(min(0.99, float(os.environ.get("FAKE_EVAL_MAP50", base + 0.15))), 4),
        "mAP50-95": round(base + 0.02, 4),
        "fitness": round(base + 0.05, 4),
    }
    per_class = [
        {"index": i, "name": name, "instances": 2,
         "precision": overall["precision"], "recall": overall["recall"],
         "f1": overall["f1"], "ap50": overall["mAP50"], "ap50_95": overall["mAP50-95"]}
        for i, name in names.items()
    ]

    for name in ARTIFACTS:
        (run_dir / name).write_bytes(b"\xff\xd8\xff\xe0fake-eval-image")

    if exit_code == 0:
        result = {
            "ok": True,
            "error": "",
            "split": spec.get("split", "val"),
            "task": spec.get("task", "detect"),
            "weights": spec.get("weights", ""),
            "data_yaml": spec.get("data_yaml", ""),
            "model_name": Path(str(spec.get("weights", ""))).name,
            "job_id": spec.get("job_id", ""),
            "tag": spec.get("tag", ""),
            "eval_id": spec.get("name", ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": delay,
            "overall": overall,
            "per_class": per_class,
            "speed": {"preprocess": 0.1, "inference": 7.2, "loss": 0.0, "postprocess": 0.3},
            "confusion_matrix": {
                "axis": "rows=真实,cols=预测",
                "labels": ["background", "square", "circle"],
                "matrix": [[0, 1, 0], [0, 2, 0], [0, 0, 1]],
            },
            "artifacts": list(ARTIFACTS),
        }
        (run_dir / "eval_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if exit_code != 0:
        # 模拟评估失败：写一份 ok=false 的结果后以非零码退出
        failed = {
            "ok": False,
            "error": "RuntimeError: fake evaluation failed",
            "split": spec.get("split", "val"),
            "task": spec.get("task", "detect"),
            "weights": spec.get("weights", ""),
            "data_yaml": spec.get("data_yaml", ""),
            "model_name": Path(str(spec.get("weights", ""))).name,
            "job_id": spec.get("job_id", ""),
            "tag": spec.get("tag", ""),
            "eval_id": spec.get("name", ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": delay,
            "overall": {},
            "per_class": [],
            "speed": {},
            "confusion_matrix": None,
            "artifacts": list(ARTIFACTS),
        }
        (run_dir / "eval_result.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("fake evaluator done", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
