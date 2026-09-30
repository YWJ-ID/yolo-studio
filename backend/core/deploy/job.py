"""导出任务的状态与可持久化记录（`job.json`）。

状态取值与训练/评估保持一致，前端可复用同一套映射。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_STOPPING = "stopping"
STATUS_STOPPED = "stopped"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

ALL_STATUSES = (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_STOPPING,
    STATUS_STOPPED,
    STATUS_FINISHED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
)
TERMINAL_STATUSES = (STATUS_STOPPED, STATUS_FINISHED, STATUS_FAILED, STATUS_INTERRUPTED)
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_STOPPING)

STATUS_LABELS = {
    STATUS_PENDING: "排队中",
    STATUS_RUNNING: "导出中",
    STATUS_STOPPING: "停止中",
    STATUS_STOPPED: "已停止",
    STATUS_FINISHED: "已完成",
    STATUS_FAILED: "失败",
    STATUS_INTERRUPTED: "已中断",
}

JOB_FILE = "job.json"


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def is_active(status: str) -> bool:
    return status in ACTIVE_STATUSES


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_deploy_id(tag: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(tag)).strip("_")[:24]
    prefix = f"deploy_{slug}" if slug else "deploy"
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


@dataclass
class DeployJob:
    """一个导出任务的持久化记录。"""

    id: str
    status: str = STATUS_PENDING
    spec: Dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""
    created_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    pid: Optional[int] = None
    returncode: Optional[int] = None
    error: str = ""
    message: str = ""
    log_count: int = 0
    stopped_by_user: bool = False
    # 结果摘要（完整结果在 deploy_result.json）
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "status_label": self.status_label,
            "spec": self.spec,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "pid": self.pid,
            "returncode": self.returncode,
            "error": self.error,
            "message": self.message,
            "log_count": self.log_count,
            "stopped_by_user": self.stopped_by_user,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeployJob":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def save_job(job: DeployJob, run_dir) -> Path:
    path = Path(run_dir) / JOB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_jobs(deploys_dir) -> List[DeployJob]:
    base = Path(deploys_dir)
    if not base.is_dir():
        return []
    jobs: List[DeployJob] = []
    for job_file in sorted(base.glob(f"*/{JOB_FILE}")):
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        job = DeployJob.from_dict(data)
        if not job.run_dir:
            job.run_dir = str(job_file.parent)
        jobs.append(job)
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)
