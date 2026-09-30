"""评估子进程入口。

    python -m core.eval.runner --spec <eval_dir>/eval_spec.json

与训练子进程同样的隔离策略：评估跑在独立进程里，失败不会影响 API。
退出码即评估成败；失败时也会写出一份 ok=false 的 eval_result.json，便于前端展示原因。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .spec import BACKEND_ULTRALYTICS, EvalSpec


def _load_spec(path) -> EvalSpec:
    return EvalSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run_spec(spec: EvalSpec) -> int:
    problems = spec.validate()
    if problems:
        for p in problems:
            print(f"[参数错误] {p}", file=sys.stderr)
        return 2

    if spec.backend == BACKEND_ULTRALYTICS:
        from .ultralytics_eval import run

        return run(spec)

    print(f"[错误] 未实现的评估后端: {spec.backend}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="core.eval.runner", description="YOLO Studio 评估子进程")
    parser.add_argument("--spec", required=True, help="eval_spec.json 路径")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"[错误] 评估参数文件不存在: {spec_path}", file=sys.stderr)
        return 2

    spec = _load_spec(spec_path)
    try:
        return run_spec(spec)
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
