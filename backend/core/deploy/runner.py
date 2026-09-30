"""导出子进程入口。

    python -m core.deploy.runner --spec <deploy_dir>/deploy_spec.json

与训练/评估同样的隔离策略。退出码：0 = 至少一个格式成功，1 = 全部失败，2 = 参数错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .spec import DeploySpec


def _load_spec(path) -> DeploySpec:
    return DeploySpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run_spec(spec: DeploySpec) -> int:
    problems = spec.validate()
    if problems:
        for p in problems:
            print(f"[参数错误] {p}", file=sys.stderr)
        return 2

    # 不支持的格式提前拦下，给出可操作的原因（缺哪个包）
    from .formats import unavailable

    for fmt in spec.normalized_formats():
        reason = unavailable(fmt)
        if reason:
            print(f"[格式不可用] {fmt}: {reason}", file=sys.stderr)
            return 2

    from .ultralytics_export import run

    return run(spec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="core.deploy.runner", description="YOLO Studio 导出子进程")
    parser.add_argument("--spec", required=True, help="deploy_spec.json 路径")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"[错误] 导出参数文件不存在: {spec_path}", file=sys.stderr)
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
