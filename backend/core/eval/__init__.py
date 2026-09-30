"""评估模块：对已训练权重在 val / test 划分上做评估并保存结构化结果。

与训练模块同构：`EvalManager` 负责调度，评估跑在独立子进程里，
结果落在评估目录的 `eval_result.json`。
"""

from __future__ import annotations

from .job import (
    ALL_STATUSES,
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    TERMINAL_STATUSES,
    EvalJob,
    is_active,
    is_terminal,
    load_jobs,
    make_eval_id,
    save_job,
)
from .manager import EvalJobNotFound, EvalManager, EvalStateError
from .compare import METRIC_LABELS, METRIC_ORDER, best_by, compare_eval_results
from .report import render_eval_report, write_eval_report
from .result import CM_AXIS, ClassMetrics, EvalResult, build_result, load_result
from .spec import (
    RESULT_FILE,
    SPEC_FILE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLITS,
    EvalSpec,
)

__all__ = [
    "ALL_STATUSES",
    "CM_AXIS",
    "ClassMetrics",
    "EvalJob",
    "EvalJobNotFound",
    "EvalManager",
    "EvalResult",
    "EvalSpec",
    "EvalStateError",
    "METRIC_LABELS",
    "METRIC_ORDER",
    "RESULT_FILE",
    "SPEC_FILE",
    "SPLITS",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "STATUS_FAILED",
    "STATUS_FINISHED",
    "STATUS_INTERRUPTED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "STATUS_STOPPING",
    "TERMINAL_STATUSES",
    "best_by",
    "build_result",
    "compare_eval_results",
    "is_active",
    "is_terminal",
    "load_jobs",
    "load_result",
    "make_eval_id",
    "render_eval_report",
    "save_job",
    "write_eval_report",
]
