"""测试用假训练进程。

模拟 ultralytics 的外部行为：逐轮追加 results.csv、打印日志、产出过程图像与
weights/last.pt。这样调度、指标解析、日志流、产物浏览都能被真实执行验证，
而不必在 CPU 上跑真正的 YOLO 训练。

环境变量（全部可选）：
    FAKE_EPOCHS  总轮数（默认 3）
    FAKE_DELAY   每轮耗时秒（默认 0.2）
    FAKE_WORK    收尾前的额外耗时秒，用于制造"长时间运行"以便测试停止（默认 0）
    FAKE_EXIT    退出码（默认 0）
    FAKE_LOG     每轮额外打印的日志行数（默认 2）

断点续训时（train_spec.json 里 resume=true）会接着已有 results.csv 的最大 epoch 继续。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0"
)


def _max_epoch(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    top = 0
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        cell = line.split(",")[0].strip()
        try:
            top = max(top, int(float(cell)))
        except ValueError:
            continue
    return top


def main() -> int:
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    run_dir = spec_path.parent

    epochs = int(os.environ.get("FAKE_EPOCHS", "3"))
    delay = float(os.environ.get("FAKE_DELAY", "0.2"))
    work = float(os.environ.get("FAKE_WORK", "0"))
    exit_code = int(os.environ.get("FAKE_EXIT", "0"))
    log_lines = int(os.environ.get("FAKE_LOG", "2"))

    csv_path = run_dir / "results.csv"
    start = _max_epoch(csv_path) + 1
    if not csv_path.is_file():
        csv_path.write_text(HEADER + "\n", encoding="utf-8")

    print(f"fake trainer start  resume={bool(spec.get('resume'))}", flush=True)
    for epoch in range(start, start + epochs):
        time.sleep(delay)
        row = (
            f"{epoch},{epoch * 10},"
            f"{2.5 / epoch:.4f},{1.5 / epoch:.4f},{1.0 / epoch:.4f},"
            f"{min(0.99, 0.2 * epoch):.4f},{min(0.99, 0.15 * epoch):.4f},"
            f"{min(0.99, 0.25 * epoch):.4f},{min(0.99, 0.1 * epoch):.4f},"
            f"{2.0 / epoch:.4f},{1.2 / epoch:.4f},{0.8 / epoch:.4f},0.01"
        )
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(row + "\n")
        print(f"Epoch {epoch}/{start + epochs - 1}  done", flush=True)
        for i in range(log_lines):
            print(f"  fake log line {i} for epoch {epoch}", flush=True)

    if work:
        print("fake trainer working...", flush=True)
        time.sleep(work)

    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"fake-last")
    (run_dir / "weights" / "best.pt").write_bytes(b"fake-best")
    for name in (
        "train_batch0.jpg",
        "train_batch1.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "results.png",
        "confusion_matrix.png",
    ):
        (run_dir / name).write_bytes(b"\xff\xd8\xff\xe0fake-image")

    print("fake trainer done", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
