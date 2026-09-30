"""部署导出模块：把训练好的权重导出为可部署格式并管理产物。

M4-01 ONNX / TorchScript，M4-02 RKNN（依赖缺失时给出明确原因），M4-03 产物校验。
"""

from __future__ import annotations

from .formats import (
    DEFAULT_FORMATS,
    FORMATS,
    ExportFormat,
    capability_report,
    cuda_available,
    get_format,
    is_known,
    unavailable,
)
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
    DeployJob,
    is_active,
    is_terminal,
    load_jobs,
    make_deploy_id,
    save_job,
)
from .manager import DeployJobNotFound, DeployManager, DeployStateError
from .result import (
    Artifact,
    DeployResult,
    VerifyReport,
    dir_size,
    load_result,
    sha256_file,
    verify_result,
)
from .spec import LOG_FILE, RESULT_FILE, SPEC_FILE, DeploySpec

__all__ = [
    "ALL_STATUSES",
    "Artifact",
    "DEFAULT_FORMATS",
    "DeployJob",
    "DeployJobNotFound",
    "DeployManager",
    "DeployResult",
    "DeploySpec",
    "DeployStateError",
    "FORMATS",
    "ExportFormat",
    "LOG_FILE",
    "RESULT_FILE",
    "SPEC_FILE",
    "STATUS_FAILED",
    "STATUS_FINISHED",
    "STATUS_INTERRUPTED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "STATUS_STOPPING",
    "TERMINAL_STATUSES",
    "VerifyReport",
    "capability_report",
    "cuda_available",
    "dir_size",
    "get_format",
    "is_active",
    "is_known",
    "is_terminal",
    "load_jobs",
    "load_result",
    "make_deploy_id",
    "save_job",
    "sha256_file",
    "unavailable",
    "verify_result",
]
