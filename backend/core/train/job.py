"""训练任务的状态机与可持久化记录。

状态流转（单向为主，允许失败/中断后 resume 回到 running）：

    pending ──start──> running ──成功──> finished
                          │  └──失败──> failed
                          │  └──stop──> stopping ──> stopped
                          └─进程失联（服务重启）──> interrupted
    finished / failed / stopped / interrupted ──resume──> running

`job.json` 写在训练目录下，服务重启后据此恢复任务列表；
正在运行但进程已不存在的任务会被标记为 interrupted，并允许续训。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 状态取值
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

# 已经结束、不会再有新进展的状态
TERMINAL_STATUSES = (STATUS_STOPPED, STATUS_FINISHED, STATUS_FAILED, STATUS_INTERRUPTED)
# 进程应当还活着的状态
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_STOPPING)

JOB_FILE = "job.json"

STATUS_LABELS = {
    STATUS_PENDING: "排队中",
    STATUS_RUNNING: "训练中",
    STATUS_STOPPING: "停止中",
    STATUS_STOPPED: "已停止",
    STATUS_FINISHED: "已完成",
    STATUS_FAILED: "失败",
    STATUS_INTERRUPTED: "已中断",
}


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def is_active(status: str) -> bool:
    return status in ACTIVE_STATUSES


@dataclass
class TrainingJob:
    """一个训练任务的持久化记录。"""

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

    # 进度摘要（由调度层随指标更新，便于列表页直接展示）
    current_epoch: int = 0
    epochs_total: int = 0
    progress: float = 0.0
    best: Dict[str, Any] = field(default_factory=dict)
    metrics_rows: int = 0
    log_count: int = 0
    attempts: int = 0
    # 手动停止过 / 被中断过，列表页需要区分
    stopped_by_user: bool = False

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
            "current_epoch": self.current_epoch,
            "epochs_total": self.epochs_total,
            "progress": self.progress,
            "best": self.best,
            "metrics_rows": self.metrics_rows,
            "log_count": self.log_count,
            "attempts": self.attempts,
            "stopped_by_user": self.stopped_by_user,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingJob":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_job_id(prefix: str = "train") -> str:
    """任务 id 同时也是训练目录名，因此需要时间戳 + 随机后缀保证唯一。"""
    import uuid

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def save_job(job: TrainingJob, run_dir) -> Path:
    import json

    path = Path(run_dir) / JOB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_jobs(runs_dir) -> List[TrainingJob]:
    """扫描训练根目录，恢复全部任务记录。"""
    import json

    base = Path(runs_dir)
    if not base.is_dir():
        return []
    jobs: List[TrainingJob] = []
    for job_file in sorted(base.glob(f"*/{JOB_FILE}")):
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        job = TrainingJob.from_dict(data)
        if not job.run_dir:
            job.run_dir = str(job_file.parent)
        jobs.append(job)
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)
