"""训练模块：调度、指标解析、资源监控、产物浏览。

对外入口是 `TrainingManager`；`TrainerBackend` 是预留的训练框架接口。
"""

from __future__ import annotations

from .artifacts import list_artifacts, pick_preview, resolve_artifact
from .backend import TrainerBackend, interpreter
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
    TrainingJob,
    is_active,
    is_terminal,
    load_jobs,
    make_job_id,
    save_job,
)
from .manager import JobNotFound, JobStateError, TrainingManager, write_spec
from .metrics import HEADLINE_METRICS, MetricRow, MetricsSeries, parse_results_csv
from .resources import ResourceMonitor, default_monitor, resolve_device
from .spec import (
    BACKEND_ULTRALYTICS,
    DEFAULT_WEIGHTS,
    TASK_CLASSIFY,
    TASK_DETECT,
    TASKS,
    TrainSpec,
)
from .ultralytics_backend import UltralyticsBackend
from .weights import (
    FORMAT_PT,
    FORMAT_STRUCTURE,
    MAX_WEIGHT_BYTES,
    candidate_weights,
    normalize_weight_filename,
    weight_format,
)

__all__ = [
    "ALL_STATUSES",
    "BACKEND_ULTRALYTICS",
    "DEFAULT_WEIGHTS",
    "FORMAT_PT",
    "FORMAT_STRUCTURE",
    "HEADLINE_METRICS",
    "MAX_WEIGHT_BYTES",
    "JobNotFound",
    "JobStateError",
    "MetricRow",
    "MetricsSeries",
    "ResourceMonitor",
    "STATUS_FAILED",
    "STATUS_FINISHED",
    "STATUS_INTERRUPTED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "STATUS_STOPPING",
    "TASKS",
    "TASK_CLASSIFY",
    "TASK_DETECT",
    "TERMINAL_STATUSES",
    "TrainSpec",
    "TrainerBackend",
    "TrainingJob",
    "TrainingManager",
    "UltralyticsBackend",
    "candidate_weights",
    "default_monitor",
    "interpreter",
    "is_active",
    "is_terminal",
    "list_artifacts",
    "load_jobs",
    "make_job_id",
    "normalize_weight_filename",
    "parse_results_csv",
    "pick_preview",
    "resolve_artifact",
    "resolve_device",
    "save_job",
    "weight_format",
    "write_spec",
]
