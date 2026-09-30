"""训练调度：子进程生命周期、状态推进、指标/日志/资源采集与事件推送。

设计要点：
  * 训练跑在 subprocess 里（`core.train.runner`），API 进程只做监管；
    训练崩溃或 OOM 不会拖垮接口服务。
  * 指标只从训练目录的 results.csv 读（`backend.parse_metrics`），不 hook ultralytics 内部。
  * 日志、指标、资源三类变化由**单个监控线程**统一采集后推送，
    避免多线程各自回调造成事件乱序。
  * 事件推送用一个同步回调列表。core 不认识 asyncio / WebSocket，
    API 层订阅后自行把事件转投到事件循环——这样 core 仍可被 CLI 直接使用。
  * 服务重启后：子进程仍在的会被"接管"（继续读 results.csv），
    已消失的标记为 interrupted 并允许续训。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .artifacts import list_artifacts, pick_preview, resolve_artifact
from .backend import TrainerBackend
from ..process import kill_tree, pid_alive
from .job import (
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    TrainingJob,
    is_active,
    is_terminal,
    load_jobs,
    make_job_id,
    now_iso,
    save_job,
)
from .metrics import MetricsSeries
from .resources import ResourceMonitor
from .spec import DEFAULT_WEIGHTS, TrainSpec
from .ultralytics_backend import UltralyticsBackend

# backend 目录：训练子进程的工作目录（core 包所在处）
BACKEND_DIR = Path(__file__).resolve().parents[2]

# 子进程 stdout 读取块大小
_READ_CHUNK = 4096
_LINE_SPLIT = re.compile(r"[\r\n]")


class JobNotFound(KeyError):
    """任务不存在。"""


class JobStateError(RuntimeError):
    """任务当前状态不允许该操作（例如对已结束的任务调用 stop）。"""


@dataclass
class JobRuntime:
    """任务的运行时状态（不持久化：进程句柄、日志缓冲、指标序列）。"""

    job: TrainingJob
    spec: TrainSpec
    process: Optional[subprocess.Popen] = None
    adopted_pid: Optional[int] = None
    logs: Deque[Tuple[int, str]] = field(default_factory=deque)
    log_total: int = 0
    series: MetricsSeries = field(default_factory=MetricsSeries)
    resources: Dict[str, Any] = field(default_factory=dict)
    emitted_last_epoch: int = 0
    emitted_log_seq: int = 0
    emitted_status: str = ""
    finalized: bool = False
    log_handle: Optional[Any] = None

    @property
    def run_dir(self) -> Path:
        return Path(self.job.run_dir)

    def add_log(self, text: str, max_lines: int) -> int:
        """追加一行日志，返回它的序号。超出上限时丢弃最旧的。"""
        seq = self.log_total
        self.logs.append((seq, text))
        self.log_total += 1
        while len(self.logs) > max_lines:
            self.logs.popleft()
        return seq


class TrainingManager:
    """管理全部训练任务。线程安全。"""

    def __init__(
        self,
        runs_dir,
        backend: Optional[TrainerBackend] = None,
        python: str = "",
        device: str = "cpu",
        poll_interval: float = 0.7,
        max_log_lines: int = 5000,
        weights_dir: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        autoload: bool = True,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.backend = backend or UltralyticsBackend()
        self.python = python
        self.device = device
        self.poll_interval = max(0.1, float(poll_interval))
        self.max_log_lines = max(100, int(max_log_lines))
        self.weights_dir = Path(weights_dir) if weights_dir else None
        # 追加到训练子进程环境变量（如 OMP_NUM_THREADS、CUDA_VISIBLE_DEVICES）
        self.extra_env: Dict[str, str] = dict(extra_env or {})

        self._lock = threading.RLock()
        self._runtimes: Dict[str, JobRuntime] = {}
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        # 任务结束（无论成败）时的回调，用于「训练完成自动评估」这类联动
        self._on_finish: List[Callable[[TrainingJob], None]] = []
        self._resources = ResourceMonitor()
        self._stop_event = threading.Event()
        self._closed = False
        self._monitor: Optional[threading.Thread] = None

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if autoload:
            self._load_existing()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="train-monitor", daemon=True
        )
        self._monitor.start()

    # =================================================================
    # 任务查询
    # =================================================================

    def list(self) -> List[TrainingJob]:
        self.refresh()
        with self._lock:
            jobs = [rt.job for rt in self._runtimes.values()]
        return sorted(jobs, key=lambda j: (j.created_at, j.id), reverse=True)

    def get(self, job_id: str) -> TrainingJob:
        with self._lock:
            rt = self._runtimes.get(job_id)
        if rt is None:
            # 允许通过 CLI 等方式在服务启动后新增的训练任务被看到
            self.refresh()
            with self._lock:
                rt = self._runtimes.get(job_id)
        if rt is None:
            raise JobNotFound(f"训练任务不存在: {job_id}")
        return rt.job

    def refresh(self) -> int:
        """扫描训练根目录，把内存里还没有的任务加载进来。返回新增数量。"""
        with self._lock:
            known = set(self._runtimes)
        added = 0
        for job in load_jobs(self.runs_dir):
            if job.id in known:
                continue
            try:
                self._load_one(job)
                added += 1
            except Exception:
                continue
        return added

    def _runtime(self, job_id: str) -> JobRuntime:
        with self._lock:
            rt = self._runtimes.get(job_id)
        if rt is None:
            raise JobNotFound(f"训练任务不存在: {job_id}")
        return rt

    def metrics(self, job_id: str, since_epoch: Optional[int] = None, max_rows: int = 0) -> MetricsSeries:
        rt = self._runtime(job_id)
        series = rt.series
        if since_epoch is not None:
            return MetricsSeries(
                columns=series.columns,
                rows=[r for r in series.rows if r.epoch > since_epoch],
            )
        if max_rows > 0 and len(series.rows) > max_rows:
            return MetricsSeries(columns=series.columns, rows=series.rows[-max_rows:])
        return series

    def logs(self, job_id: str, offset: int = 0, limit: int = 500) -> Dict[str, Any]:
        rt = self._runtime(job_id)
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        lines = [{"seq": seq, "text": text} for seq, text in rt.logs if seq >= offset][:limit]
        return {"lines": lines, "total": rt.log_total, "next_offset": offset + len(lines)}

    def resources(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        if job_id is None:
            return self._resources.sample(None)
        rt = self._runtime(job_id)
        if rt.resources:
            return rt.resources
        return self._resources.sample(rt.job.pid or rt.adopted_pid)

    def capabilities(self) -> Dict[str, Any]:
        return self._resources.capabilities()

    def artifacts(self, job_id: str) -> Dict[str, Any]:
        rt = self._runtime(job_id)
        data = list_artifacts(rt.run_dir)
        data["preview"] = pick_preview(rt.run_dir)
        return data

    def artifact_path(self, job_id: str, name: str):
        rt = self._runtime(job_id)
        return resolve_artifact(rt.run_dir, name)

    # =================================================================
    # 事件订阅
    # =================================================================

    def subscribe(self, job_id: str, callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        """订阅某任务的增量事件。返回取消订阅的函数。"""
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                subs = self._subscribers.get(job_id)
                if subs and callback in subs:
                    subs.remove(callback)
                if subs is not None and not subs:
                    self._subscribers.pop(job_id, None)

        return unsubscribe

    def _emit(self, job_id: str, event: Dict[str, Any]) -> None:
        event.setdefault("job_id", job_id)
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        for cb in subs:
            try:
                cb(event)
            except Exception:
                # 单个订阅者出错不能影响其它订阅者和监控线程
                pass

    def on_finish(self, callback: Callable[["TrainingJob"], None]) -> Callable[[], None]:
        """注册任务结束回调（成功或失败都会调用）。返回取消注册的函数。

        用于「训练完成自动评估」这类联动：core 不知道评估的存在，
        由上层在回调里决定做什么，core 只负责在恰当的时候通知。
        """
        with self._lock:
            if callback in self._on_finish:
                return lambda: None
            self._on_finish.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._on_finish:
                    self._on_finish.remove(callback)

        return unsubscribe

    def _notify_finish(self, job: TrainingJob) -> None:
        with self._lock:
            callbacks = list(self._on_finish)
        for cb in callbacks:
            try:
                cb(job)
            except Exception:
                # 回调失败不能影响任务收尾
                pass

    def snapshot(self, job_id: str, log_limit: int = 200) -> Dict[str, Any]:
        rt = self._runtime(job_id)
        logs = self.logs(job_id, offset=max(0, rt.log_total - log_limit), limit=log_limit)
        return {
            "type": "snapshot",
            "job_id": job_id,
            "job": rt.job.to_dict(),
            "metrics": rt.series.to_dict(),
            "logs": logs,
            "resources": rt.resources,
            "monitor": self._resources.capabilities(),
        }

    # =================================================================
    # 启动 / 停止 / 续训
    # =================================================================

    def start(self, spec: TrainSpec) -> TrainingJob:
        """创建并启动一个训练任务。"""
        self._apply_defaults(spec)
        problems = spec.validate()
        if problems:
            raise ValueError("；".join(problems))

        run_dir = spec.run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"训练目录非空: {run_dir}")

        job = TrainingJob(
            id=spec.name,
            status=STATUS_PENDING,
            spec=spec.to_dict(),
            run_dir=str(run_dir),
            created_at=now_iso(),
            epochs_total=spec.epochs,
        )
        rt = JobRuntime(job=job, spec=spec)
        with self._lock:
            self._runtimes[job.id] = rt

        self._spawn(rt)
        return job

    def stop(self, job_id: str) -> TrainingJob:
        rt = self._runtime(job_id)
        if not is_active(rt.job.status):
            raise JobStateError(f"任务当前状态为 {rt.job.status}，无法停止")

        rt.job.stopped_by_user = True
        rt.job.status = STATUS_STOPPING
        rt.job.message = "正在停止训练进程"
        self._persist(rt)
        self._emit_status(rt)

        if rt.process is not None:
            self._kill_tree(rt.process.pid)
        elif rt.adopted_pid:
            self._kill_tree(rt.adopted_pid)

        self._finalize(rt, rt.process.poll() if rt.process else None)
        return rt.job

    def resume(self, job_id: str, **overrides: Any) -> TrainingJob:
        """从 last.pt 断点续训。沿用原训练目录，指标按 epoch 去重后继续累积。"""
        rt = self._runtime(job_id)
        if not is_terminal(rt.job.status):
            raise JobStateError(f"任务当前状态为 {rt.job.status}，只有已结束的任务可以续训")

        last = self.backend.last_weights(rt.run_dir)
        if last is None:
            raise JobStateError(f"找不到断点权重 weights/last.pt，无法续训: {rt.run_dir}")

        spec = TrainSpec.from_dict(rt.job.spec)
        spec.resume = True
        for key, value in overrides.items():
            if value is not None and hasattr(spec, key):
                setattr(spec, key, value)

        rt.spec = spec
        rt.finalized = False
        rt.job.spec = spec.to_dict()
        rt.job.status = STATUS_PENDING
        rt.job.returncode = None
        rt.job.error = ""
        rt.job.ended_at = ""
        rt.job.stopped_by_user = False
        rt.job.message = f"从 {last.name} 续训"

        self._spawn(rt)
        return rt.job

    # =================================================================
    # 内部：参数与启动
    # =================================================================

    def _apply_defaults(self, spec: TrainSpec) -> None:
        if not spec.python:
            spec.python = self.python
        if not spec.device or spec.device == "auto":
            spec.device = self.device
        if not spec.weights:
            spec.weights = DEFAULT_WEIGHTS.get(spec.task, DEFAULT_WEIGHTS["detect"])
        if not spec.name:
            spec.name = make_job_id()
        if not spec.project:
            spec.project = str(self.runs_dir)
        # 必须是绝对路径：ultralytics 对相对 project 会拼上它自己的 SETTINGS["runs_dir"]，
        # 那样 results.csv 会写到别处，指标就再也读不到。
        spec.project = str(Path(spec.project).expanduser().resolve())

    def _spawn(self, rt: JobRuntime) -> None:
        spec = rt.spec
        rt.run_dir.mkdir(parents=True, exist_ok=True)

        # 权重解析在这里做一次，日志与消息里能看到实际用的是哪个
        try:
            resolved = self.backend.resolve_weights(spec.weights, base_dir=self.weights_dir)
            rt.job.message = f"权重: {resolved}"
        except ValueError as exc:
            rt.job.status = STATUS_FAILED
            rt.job.error = str(exc)
            rt.job.ended_at = now_iso()
            self._persist(rt)
            self._emit_status(rt)
            raise

        write_spec(spec)

        command = self.backend.build_command(spec)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self.extra_env)

        rt.job.attempts += 1
        rt.job.started_at = now_iso()
        rt.job.ended_at = ""
        rt.job.returncode = None
        rt.job.error = ""
        rt.job.pid = None
        rt.adopted_pid = None
        rt.finalized = False

        try:
            rt.process = subprocess.Popen(
                command,
                cwd=str(BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )
        except Exception as exc:
            rt.job.status = STATUS_FAILED
            rt.job.error = f"启动训练进程失败: {exc}"
            rt.job.ended_at = now_iso()
            self._persist(rt)
            self._emit_status(rt)
            return

        rt.job.pid = rt.process.pid
        rt.job.status = STATUS_RUNNING
        self._open_log_file(rt)
        self._append_log(rt, f"[studio] 启动训练进程 pid={rt.process.pid}")
        self._append_log(rt, f"[studio] 命令: {' '.join(command)}")

        self._persist(rt)
        self._emit_status(rt)

        reader = threading.Thread(target=self._pump_logs, args=(rt,), daemon=True)
        reader.start()

    def _open_log_file(self, rt: JobRuntime) -> None:
        """日志同时落盘到 run_dir/train.log，服务重启或事后查看都还能看到。"""
        if rt.log_handle is not None:
            return
        try:
            rt.log_handle = open(rt.run_dir / "train.log", "a", encoding="utf-8")
        except Exception:
            rt.log_handle = None

    def _append_log(self, rt: JobRuntime, text: str) -> None:
        rt.add_log(text, self.max_log_lines)
        if rt.log_handle is not None:
            try:
                rt.log_handle.write(text + "\n")
                rt.log_handle.flush()
            except Exception:
                rt.log_handle = None

    def _pump_logs(self, rt: JobRuntime) -> None:
        """读取子进程 stdout/stderr，按 \\r 与 \\n 切分成行。

        训练框架的进度条用 \\r 刷新，若只按 \\n 切分会攒成超长行。
        """
        proc = rt.process
        if proc is None or proc.stdout is None:
            return
        buffer = ""
        try:
            while True:
                chunk = proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", "replace")
                parts = _LINE_SPLIT.split(buffer)
                buffer = parts.pop()
                for part in parts:
                    text = part.rstrip()
                    if text:
                        self._append_log(rt, text)
        except Exception:
            pass
        finally:
            if buffer.strip():
                self._append_log(rt, buffer.strip())
            try:
                proc.stdout.close()
            except Exception:
                pass

    # =================================================================
    # 内部：监控线程
    # =================================================================

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    runtimes = list(self._runtimes.values())
                for rt in runtimes:
                    try:
                        self._tick(rt)
                    except Exception:
                        pass
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval)

    def _tick(self, rt: JobRuntime) -> None:
        if is_active(rt.job.status):
            if not self._alive(rt):
                code = rt.process.poll() if rt.process is not None else None
                self._finalize(rt, code)
            else:
                self._refresh_metrics(rt)
                self._refresh_resources(rt)
        elif not rt.finalized:
            # 已结束的任务补一次指标，确保最后一轮不丢
            self._refresh_metrics(rt)

        self._flush_logs(rt)

    def _alive(self, rt: JobRuntime) -> bool:
        if rt.process is not None:
            return rt.process.poll() is None
        if rt.adopted_pid:
            return _pid_alive(rt.adopted_pid)
        return False

    def _refresh_metrics(self, rt: JobRuntime) -> None:
        csv_path = self.backend.results_csv(rt.run_dir)
        series = self.backend.parse_metrics(csv_path)
        if not series.rows:
            return

        new_rows = [r for r in series.rows if r.epoch > rt.emitted_last_epoch]
        rt.series = series

        rt.job.current_epoch = series.last_epoch or 0
        rt.job.metrics_rows = series.epochs
        total = rt.job.epochs_total or rt.spec.epochs or 0
        if total:
            rt.job.progress = round(min(1.0, series.epochs / total), 4)
        best = series.best_headline()
        if best:
            rt.job.best = best

        if new_rows:
            rt.emitted_last_epoch = max(r.epoch for r in new_rows)
            self._persist(rt)
            self._emit(
                rt.job.id,
                {
                    "type": "metrics",
                    "rows": [r.to_dict() for r in new_rows],
                    "progress": series.progress(total),
                    "best": rt.job.best,
                },
            )

    def _refresh_resources(self, rt: JobRuntime) -> None:
        sample = self._resources.sample(rt.job.pid or rt.adopted_pid)
        rt.resources = sample
        self._emit(rt.job.id, {"type": "resources", "sample": sample})

    def _flush_logs(self, rt: JobRuntime) -> None:
        if rt.emitted_log_seq >= rt.log_total:
            return
        pending = [(seq, text) for seq, text in rt.logs if seq >= rt.emitted_log_seq]
        rt.emitted_log_seq = rt.log_total
        if not pending:
            return
        self._emit(
            rt.job.id,
            {
                "type": "log",
                "lines": [{"seq": seq, "text": text} for seq, text in pending],
                "total": rt.log_total,
            },
        )

    def _finalize(self, rt: JobRuntime, returncode: Optional[int]) -> None:
        if rt.finalized:
            return
        rt.finalized = True

        # 等读日志的线程把剩余输出写完（最多 2 秒）
        if rt.process is not None:
            try:
                rt.process.wait(timeout=2)
            except Exception:
                pass
        deadline = time.time() + 2.0
        while rt.emitted_log_seq < rt.log_total and time.time() < deadline:
            time.sleep(0.05)

        rt.job.returncode = returncode
        rt.job.pid = None
        rt.job.ended_at = now_iso()

        # 先算出终态，但**不要马上写进 job.status**：
        # 外部一旦看到 finished 就会去读指标，若此时 results.csv 还没解析完，
        # 会读到最后几轮缺失的结果。因此先刷新指标与日志，最后才置终态。
        if rt.job.stopped_by_user:
            status, message = STATUS_STOPPED, "已手动停止"
        elif returncode == 0:
            status, message = STATUS_FINISHED, "训练完成"
        elif rt.process is None:
            # 接管的任务：进程消失且没有退出码，只能按产物判断
            status, message = STATUS_INTERRUPTED, "进程已结束，未能获取退出码（服务重启期间接管的任务）"
        else:
            status, message = STATUS_FAILED, "训练失败"
            rt.job.error = rt.job.error or f"训练进程退出码 {returncode}"

        self._refresh_metrics(rt)
        if status == STATUS_FINISHED and rt.job.epochs_total:
            rt.job.progress = 1.0
        self._flush_logs(rt)

        rt.job.status = status
        rt.job.message = message
        self._persist(rt)
        self._close_log_file(rt)
        # 先让上层联动（注册模型 / 自动评估），再对外宣告 finished
        self._notify_finish(rt.job)
        self._emit_status(rt)
        self._emit(
            rt.job.id,
            {
                "type": "finished",
                "status": rt.job.status,
                "returncode": returncode,
                "artifacts": list_artifacts(rt.run_dir),
            },
        )

    def _emit_status(self, rt: JobRuntime) -> None:
        rt.emitted_status = rt.job.status
        self._emit(
            rt.job.id,
            {
                "type": "status",
                "status": rt.job.status,
                "status_label": rt.job.status_label,
                "message": rt.job.message,
                "error": rt.job.error,
                "progress": rt.job.progress,
                "current_epoch": rt.job.current_epoch,
            },
        )

    def _persist(self, rt: JobRuntime) -> None:
        try:
            save_job(rt.job, rt.run_dir)
        except Exception:
            pass

    @staticmethod
    def _close_log_file(rt: JobRuntime) -> None:
        if rt.log_handle is not None:
            try:
                rt.log_handle.close()
            except Exception:
                pass
            rt.log_handle = None

    # =================================================================
    # 内部：进程控制与恢复
    # =================================================================

    @staticmethod
    def _kill_tree(pid: int) -> None:
        """终止整个进程树（ultralytics 可能拉起 dataloader 子进程）。"""
        kill_tree(pid)

    def _load_existing(self) -> None:
        """服务启动时恢复任务列表。"""
        for job in load_jobs(self.runs_dir):
            try:
                self._load_one(job)
            except Exception:
                # 单条记录损坏不应让整个任务列表加载失败
                continue

    def _load_one(self, job: TrainingJob) -> None:
        spec = TrainSpec.from_dict(job.spec or {})
        rt = JobRuntime(job=job, spec=spec)
        # 先恢复历史日志，接管提示才会排在其后
        self._load_log_history(rt)

        if is_active(job.status):
            if job.pid and pid_alive(job.pid):
                # 进程还在：接管，继续读 results.csv（stdout 已无法接续）
                rt.adopted_pid = job.pid
                job.status = STATUS_RUNNING
                job.message = (
                    f"服务重启后接管 pid={job.pid}；日志从重启后开始记录，"
                    "指标仍会从 results.csv 继续更新"
                )
                rt.add_log("[studio] 服务重启，已接管仍在运行的训练进程", self.max_log_lines)
            else:
                job.status = STATUS_INTERRUPTED
                job.message = "服务重启时训练进程已不存在"
                job.error = job.error or "进程失联"
                job.ended_at = job.ended_at or now_iso()
                job.pid = None
            save_job(job, rt.run_dir)

        # 历史指标载入内存，列表页就能直接显示进度与最优指标
        try:
            rt.series = self.backend.parse_metrics(self.backend.results_csv(rt.run_dir))
            rt.emitted_last_epoch = rt.series.last_epoch or 0
        except Exception:
            pass

        # 已结束的任务不再需要监控线程反复刷新
        if is_terminal(job.status):
            rt.finalized = True

        with self._lock:
            self._runtimes[job.id] = rt

    def _load_log_history(self, rt: JobRuntime) -> None:
        """把 train.log 的尾部读回内存缓冲，并标记为已推送。"""
        log_path = rt.run_dir / "train.log"
        if not log_path.is_file():
            return
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        lines = text.splitlines()[-self.max_log_lines :]
        for line in lines:
            rt.add_log(line, self.max_log_lines)
        rt.emitted_log_seq = rt.log_total

    # =================================================================
    # 生命周期
    # =================================================================

    def shutdown(self, kill_jobs: bool = False) -> None:
        """停止监控线程。默认不结束正在跑的训练（服务重启不应中断训练）。"""
        if kill_jobs:
            with self._lock:
                runtimes = list(self._runtimes.values())
            for rt in runtimes:
                if is_active(rt.job.status):
                    try:
                        self.stop(rt.job.id)
                    except Exception:
                        pass
        self._closed = True
        self._stop_event.set()
        if self._monitor is not None and self._monitor.is_alive():
            self._monitor.join(timeout=3)
        self._resources.close()


def write_spec(spec: TrainSpec) -> Path:
    """把训练参数写入 run_dir/train_spec.json（子进程与事后追溯都用它）。"""
    path = spec.spec_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
