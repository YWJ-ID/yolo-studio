"""训练子进程入口。

调度层通过 `python -m core.train.runner --spec <run_dir>/train_spec.json` 启动它。
子进程与 API 进程完全隔离：训练崩溃、OOM、段错误都不会影响接口服务。

单独调试某次训练时可以直接跑这个命令；退出码即训练成败。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .spec import BACKEND_ULTRALYTICS, TrainSpec


def _load_spec(path) -> TrainSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainSpec.from_dict(data)


def run_spec(spec: TrainSpec) -> int:
    problems = spec.validate()
    if problems:
        for p in problems:
            print(f"[参数错误] {p}", file=sys.stderr)
        return 2

    if spec.backend == BACKEND_ULTRALYTICS:
        from .ultralytics_backend import train

        return train(spec)

    print(f"[错误] 未实现的训练后端: {spec.backend}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="core.train.runner", description="YOLO Studio 训练子进程")
    parser.add_argument("--spec", required=True, help="train_spec.json 路径")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"[错误] 训练参数文件不存在: {spec_path}", file=sys.stderr)
        return 2

    spec = _load_spec(spec_path)
    try:
        return run_spec(spec)
    except KeyboardInterrupt:
        return 130
    except Exception:
        # 完整堆栈写到 stderr，会被调度层收进日志，前端能看到真实原因
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
