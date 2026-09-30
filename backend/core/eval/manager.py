"""评估调度：子进程生命周期、状态推进、日志与结果采集。

与训练调度（core.train.manager）同构但更简单：
  * 没有实时指标曲线（评估是一次性的，结果在进程结束时一次性写出）；
  * 没有断点续训；
  * 服务重启后不接管进程，直接标记 interrupted（评估通常很快，重跑即可）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..process import kill_tree, pid_alive
from ..train.artifacts import list_artifacts, resolve_artifact
from ..train.resources import ResourceMonitor
from .job import (
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    EvalJob,
    is_active,
    is_terminal,
    load_jobs,
    make_eval_id,
    now_iso,
    save_job,
)
from .result import EvalResult, load_result
from .spec import EvalSpec

BACKEND_DIR = Path(__file__).resolve().parents[2]

_READ_CHUNK = 4096
_LINE_SPLIT = re.compile(r"[\r\n]")


class EvalJobNotFound(KeyError):
    """评估任务不存在。"""


class EvalStateError(RuntimeError):
    """评估任务当前状态不允许该操作。"""


@dataclass
class EvalRuntime:
    job: EvalJob
    spec: EvalSpec
    process: Optional[subprocess.Popen] = None
    # 服务重启后接管的进程（stdout 已断，只能等它结束再读结果）
    adopted_pid: Optional[int] = None
    logs: Deque[Tuple[int, str]] = field(default_factory=deque)
    log_total: int = 0
    result: Optional[EvalResult] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    emitted_log_seq: int = 0
    finalized: bool = False
    log_handle: Optional[Any] = None

    @property
    def run_dir(self) -> Path:
        return Path(self.job.run_dir)

    def add_log(self, text: str, max_lines: int) -> int:
        seq = self.log_total
        self.logs.append((seq, text))
        self.log_total += 1
        while len(self.logs) > max_lines:
            self.logs.popleft()
        return seq


class EvalManager:
    """管理全部评估任务。线程安全。"""

    def __init__(
        self,
        evals_dir,
        python: str = "",
        device: str = "cpu",
        poll_interval: float = 0.5,
        max_log_lines: int = 3000,
        extra_env: Optional[Dict[str, str]] = None,
        command_builder: Optional[Callable[[EvalSpec], List[str]]] = None,
        autoload: bool = True,
    ) -> None:
        self.evals_dir = Path(evals_dir)
        self.python = python
        self.device = device
        self.poll_interval = max(0.1, float(poll_interval))
        self.max_log_lines = max(100, int(max_log_lines))
        self.extra_env: Dict[str, str] = dict(extra_env or {})
        # 覆盖启动命令的钩子：用于测试注入假评估进程，
        # 也可用于把评估指向别的解释器/包装脚本。
        self.command_builder = command_builder

        self._lock = threading.RLock()
        self._runtimes: Dict[str, EvalRuntime] = {}
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        # 评估结束（无论成败）时的回调，用于把结果挂到模型卡片上
        self._on_finish: List[Callable[[EvalJob, Optional[EvalResult]], None]] = []
        self._resources = ResourceMonitor()
        self._stop_event = threading.Event()
        self._monitor: Optional[threading.Thread] = None

        self.evals_dir.mkdir(parents=True, exist_ok=True)
        if autoload:
            self._load_existing()
        self._monitor = threading.Thread(target=self._monitor_loop, name="eval-monitor", daemon=True)
        self._monitor.start()

    # =================================================================
    # 查询
    # =================================================================

    def list(self) -> List[EvalJob]:
        self.refresh()
        with self._lock:
            jobs = [rt.job for rt in self._runtimes.values()]
        return sorted(jobs, key=lambda j: (j.created_at, j.id), reverse=True)

    def get(self, eval_id: str) -> EvalJob:
        with self._lock:
            rt = self._runtimes.get(eval_id)
        if rt is None:
            # 允许手工执行 core.eval.runner 产生的评估被看到
            self.refresh()
            with self._lock:
                rt = self._runtimes.get(eval_id)
        if rt is None:
            raise EvalJobNotFound(f"评估任务不存在: {eval_id}")
        return rt.job

    def refresh(self) -> int:
        """扫描评估根目录，把内存里还没有的任务加载进来。返回新增数量。"""
        with self._lock:
            known = set(self._runtimes)
        added = 0
        for job in load_jobs(self.evals_dir):
            if job.id in known:
                continue
            try:
                self._load_one(job)
                added += 1
            except Exception:
                continue
        return added

    def _runtime(self, eval_id: str) -> EvalRuntime:
        with self._lock:
            rt = self._runtimes.get(eval_id)
        if rt is None:
            raise EvalJobNotFound(f"评估任务不存在: {eval_id}")
        return rt

    def result(self, eval_id: str) -> Optional[EvalResult]:
        rt = self._runtime(eval_id)
        if rt.result is None:
            rt.result = load_result(rt.run_dir)
        return rt.result

    def logs(self, eval_id: str, offset: int = 0, limit: int = 500) -> Dict[str, Any]:
        rt = self._runtime(eval_id)
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        lines = [{"seq": seq, "text": text} for seq, text in rt.logs if seq >= offset][:limit]
        return {"lines": lines, "total": rt.log_total, "next_offset": offset + len(lines)}

    def resources(self, eval_id: Optional[str] = None) -> Dict[str, Any]:
        if eval_id is None:
            return self._resources.sample(None)
        rt = self._runtime(eval_id)
        if rt.resources:
            return rt.resources
        return self._resources.sample(rt.job.pid)

    def capabilities(self) -> Dict[str, Any]:
        return self._resources.capabilities()

    def artifacts(self, eval_id: str) -> Dict[str, Any]:
        rt = self._runtime(eval_id)
        data = list_artifacts(rt.run_dir)
        return data

    def artifact_path(self, eval_id: str, name: str):
        rt = self._runtime(eval_id)
        return resolve_artifact(rt.run_dir, name)

    # =================================================================
    # 事件
    # =================================================================

    def subscribe(self, eval_id: str, callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.setdefault(eval_id, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                subs = self._subscribers.get(eval_id)
                if subs and callback in subs:
                    subs.remove(callback)
                if subs is not None and not subs:
                    self._subscribers.pop(eval_id, None)

        return unsubscribe

    def _emit(self, eval_id: str, event: Dict[str, Any]) -> None:
        event.setdefault("job_id", eval_id)
        with self._lock:
            subs = list(self._subscribers.get(eval_id, []))
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass

    def on_finish(
        self, callback: Callable[[EvalJob, Optional[EvalResult]], None]
    ) -> Callable[[], None]:
        """注册评估结束回调（成功或失败都会调用）。返回取消注册的函数。

        用于把评估结果挂到模型卡片上：core.eval 不认识模型库，
        由上层在回调里写卡片。
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

    def _notify_finish(self, job: EvalJob, result: Optional[EvalResult]) -> None:
        with self._lock:
            callbacks = list(self._on_finish)
        for cb in callbacks:
            try:
                cb(job, result)
            except Exception:
                pass

    def snapshot(self, eval_id: str, log_limit: int = 200) -> Dict[str, Any]:
        rt = self._runtime(eval_id)
        logs = self.logs(eval_id, offset=max(0, rt.log_total - log_limit), limit=log_limit)
        result = self.result(eval_id)
        return {
            "type": "snapshot",
            "job_id": eval_id,
            "job": rt.job.to_dict(),
            "result": result.to_dict() if result else None,
            "logs": logs,
            "resources": rt.resources,
            "monitor": self._resources.capabilities(),
        }

    # =================================================================
    # 启动 / 停止
    # =================================================================

    def start(self, spec: EvalSpec) -> EvalJob:
        self._apply_defaults(spec)
        problems = spec.validate()
        if problems:
            raise ValueError("；".join(problems))

        run_dir = spec.run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"评估目录非空: {run_dir}")

        job = EvalJob(
            id=spec.name,
            status=STATUS_PENDING,
            spec=spec.to_dict(),
            run_dir=str(run_dir),
            created_at=now_iso(),
        )
        rt = EvalRuntime(job=job, spec=spec)
        with self._lock:
            self._runtimes[job.id] = rt

        self._spawn(rt)
        return job

    def stop(self, eval_id: str) -> EvalJob:
        rt = self._runtime(eval_id)
        if not is_active(rt.job.status):
            raise EvalStateError(f"评估任务当前状态为 {rt.job.status}，无法停止")

        rt.job.stopped_by_user = True
        rt.job.status = STATUS_STOPPING
        rt.job.message = "正在停止评估进程"
        self._persist(rt)
        self._emit_status(rt)

        if rt.process is not None:
            kill_tree(rt.process.pid)
        elif rt.adopted_pid:
            kill_tree(rt.adopted_pid)
        self._finalize(rt, rt.process.poll() if rt.process else None)
        return rt.job

    # =================================================================
    # 内部
    # =================================================================

    def _build_command(self, spec: EvalSpec) -> List[str]:
        if self.command_builder is not None:
            return list(self.command_builder(spec))
        return [
            spec.python or sys.executable,
            "-m",
            "core.eval.runner",
            "--spec",
            str(spec.spec_path),
        ]

    def _apply_defaults(self, spec: EvalSpec) -> None:
        if not spec.python:
            spec.python = self.python
        if not spec.device or spec.device == "auto":
            spec.device = self.device
        if not spec.name:
            spec.name = make_eval_id(spec.tag)
        if not spec.project:
            spec.project = str(self.evals_dir)
        # 与训练同样的要求：绝对路径，否则 ultralytics 会拼上它自己的 runs_dir
        spec.project = str(Path(spec.project).expanduser().resolve())

    def _spawn(self, rt: EvalRuntime) -> None:
        spec = rt.spec
        rt.run_dir.mkdir(parents=True, exist_ok=True)
        _write_spec(spec)

        command = self._build_command(spec)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self.extra_env)

        rt.job.started_at = now_iso()
        rt.job.ended_at = ""
        rt.job.returncode = None
        rt.job.error = ""
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
            rt.job.error = f"启动评估进程失败: {exc}"
            rt.job.ended_at = now_iso()
            self._persist(rt)
            self._emit_status(rt)
            return

        rt.job.pid = rt.process.pid
        rt.job.status = STATUS_RUNNING
        self._open_log_file(rt)
        self._append_log(rt, f"[studio] 启动评估进程 pid={rt.process.pid}")
        self._append_log(rt, f"[studio] 命令: {' '.join(command)}")

        self._persist(rt)
        self._emit_status(rt)

        reader = threading.Thread(target=self._pump_logs, args=(rt,), daemon=True)
        reader.start()

    def _open_log_file(self, rt: EvalRuntime) -> None:
        if rt.log_handle is not None:
            return
        try:
            rt.log_handle = open(rt.run_dir / "eval.log", "a", encoding="utf-8")
        except Exception:
            rt.log_handle = None

    def _append_log(self, rt: EvalRuntime, text: str) -> None:
        rt.add_log(text, self.max_log_lines)
        if rt.log_handle is not None:
            try:
                rt.log_handle.write(text + "\n")
                rt.log_handle.flush()
            except Exception:
                rt.log_handle = None

    def _pump_logs(self, rt: EvalRuntime) -> None:
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

    def _tick(self, rt: EvalRuntime) -> None:
        if is_active(rt.job.status):
            if rt.process is not None:
                running = rt.process.poll() is None
                pid = rt.job.pid
            elif rt.adopted_pid:
                running = pid_alive(rt.adopted_pid)
                pid = rt.adopted_pid
            else:
                running = False
                pid = None

            if running:
                rt.resources = self._resources.sample(pid)
                self._emit(rt.job.id, {"type": "resources", "sample": rt.resources})
            else:
                code = rt.process.poll() if rt.process is not None else None
                self._finalize(rt, code)
        self._flush_logs(rt)

    def _flush_logs(self, rt: EvalRuntime) -> None:
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

    def _finalize(self, rt: EvalRuntime, returncode: Optional[int]) -> None:
        if rt.finalized:
            return
        rt.finalized = True

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

        result = load_result(rt.run_dir)
        rt.result = result

        # 与训练一致：先把结果算出来，最后才置终态，
        # 避免外部看到 finished 时 result/metrics 还没就绪。
        if rt.job.stopped_by_user:
            status, message = STATUS_STOPPED, "已手动停止"
        elif returncode == 0:
            status, message = STATUS_FINISHED, "评估完成"
        elif returncode is None and result is not None and result.ok:
            # 接管的任务：进程已结束、退出码拿不到，但结果文件写全了
            status, message = STATUS_FINISHED, "评估完成（服务重启后接管）"
        elif returncode is None and result is None:
            status, message = STATUS_INTERRUPTED, "评估进程已结束，未产出结果"
        else:
            status, message = STATUS_FAILED, "评估失败"
            if result is not None and result.error:
                rt.job.error = result.error
            else:
                rt.job.error = f"评估进程退出码 {returncode}"

        if result is not None:
            rt.job.metrics = {
                "split": result.split,
                "model_name": result.model_name,
                "overall": result.overall,
                "num_classes": len(result.per_class),
            }

        self._flush_logs(rt)
        rt.job.status = status
        rt.job.message = message
        self._persist(rt)
        self._close_log_file(rt)
        # 先让上层把结果挂到模型库，再对外宣告 finished：
        # 否则客户端收到 finished 后立刻取模型卡片，会读到还没更新的旧卡片。
        self._notify_finish(rt.job, result)
        self._emit_status(rt)
        self._emit(
            rt.job.id,
            {
                "type": "finished",
                "status": rt.job.status,
                "returncode": returncode,
                "result": result.to_dict() if result else None,
                "artifacts": list_artifacts(rt.run_dir),
            },
        )

    def _emit_status(self, rt: EvalRuntime) -> None:
        self._emit(
            rt.job.id,
            {
                "type": "status",
                "status": rt.job.status,
                "status_label": rt.job.status_label,
                "message": rt.job.message,
                "error": rt.job.error,
            },
        )

    def _persist(self, rt: EvalRuntime) -> None:
        try:
            save_job(rt.job, rt.run_dir)
        except Exception:
            pass

    @staticmethod
    def _close_log_file(rt: EvalRuntime) -> None:
        if rt.log_handle is not None:
            try:
                rt.log_handle.close()
            except Exception:
                pass
            rt.log_handle = None

    def _load_existing(self) -> None:
        """服务启动时恢复评估列表。评估不接管进程：仍在"运行"的直接标记中断。"""
        for job in load_jobs(self.evals_dir):
            try:
                self._load_one(job)
            except Exception:
                # 单条记录损坏不应让整个列表加载失败
                continue

    def _load_one(self, job: EvalJob) -> None:
        spec = EvalSpec.from_dict(job.spec or {})
        rt = EvalRuntime(job=job, spec=spec)
        self._load_log_history(rt)

        if is_active(job.status):
            if job.pid and pid_alive(job.pid):
                # 进程还在：接管，等它结束后读 eval_result.json（stdout 已无法接续）
                rt.adopted_pid = job.pid
                job.status = STATUS_RUNNING
                job.message = (
                    f"服务重启后接管 pid={job.pid}；结果将在进程结束后从 eval_result.json 读取"
                )
            else:
                job.status = STATUS_INTERRUPTED
                job.message = "服务重启时评估进程已不存在"
                job.error = job.error or "进程失联"
                job.ended_at = job.ended_at or now_iso()
                job.pid = None
            save_job(job, rt.run_dir)

        rt.result = load_result(rt.run_dir)
        if rt.result is not None and not job.metrics:
            job.metrics = {
                "split": rt.result.split,
                "model_name": rt.result.model_name,
                "overall": rt.result.overall,
                "num_classes": len(rt.result.per_class),
            }
        if is_terminal(job.status):
            rt.finalized = True

        with self._lock:
            self._runtimes[job.id] = rt

    def _load_log_history(self, rt: EvalRuntime) -> None:
        log_path = rt.run_dir / "eval.log"
        if not log_path.is_file():
            return
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        for line in text.splitlines()[-self.max_log_lines :]:
            rt.add_log(line, self.max_log_lines)
        rt.emitted_log_seq = rt.log_total

    def shutdown(self, kill_jobs: bool = False) -> None:
        if kill_jobs:
            with self._lock:
                runtimes = list(self._runtimes.values())
            for rt in runtimes:
                if is_active(rt.job.status):
                    try:
                        self.stop(rt.job.id)
                    except Exception:
                        pass
        self._stop_event.set()
        if self._monitor is not None and self._monitor.is_alive():
            self._monitor.join(timeout=3)
        self._resources.close()


def _write_spec(spec: EvalSpec) -> Path:
    path = spec.spec_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
