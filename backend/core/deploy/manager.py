"""导出调度：子进程生命周期、状态推进、日志与结果采集。

结构与其他调度器一致，但导出的终态判定依赖 deploy_result.json：
只要有一个格式成功即 finished，全部失败才 failed。
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
from ..train.resources import ResourceMonitor
from .formats import capability_report, unavailable
from .job import (
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    DeployJob,
    is_active,
    is_terminal,
    load_jobs,
    make_deploy_id,
    now_iso,
    save_job,
)
from .result import DeployResult, load_result, verify_result
from .spec import DeploySpec

BACKEND_DIR = Path(__file__).resolve().parents[2]

_READ_CHUNK = 4096
_LINE_SPLIT = re.compile(r"[\r\n]")


class DeployJobNotFound(KeyError):
    """导出任务不存在。"""


class DeployStateError(RuntimeError):
    """导出任务当前状态不允许该操作。"""


@dataclass
class DeployRuntime:
    job: DeployJob
    spec: DeploySpec
    process: Optional[subprocess.Popen] = None
    adopted_pid: Optional[int] = None
    logs: Deque[Tuple[int, str]] = field(default_factory=deque)
    log_total: int = 0
    result: Optional[DeployResult] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    emitted_log_seq: int = 0
    finalized: bool = False
    log_handle: Optional[Any] = None
    # `_finalize` 执行完毕才置位：让 stop() 等它跑完，避免返回一个仍停在中间态的任务。
    finalize_done: Optional[threading.Event] = None

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


class DeployManager:
    """管理全部导出任务。线程安全。"""

    def __init__(
        self,
        deploys_dir,
        python: str = "",
        device: str = "cpu",
        poll_interval: float = 0.5,
        max_log_lines: int = 3000,
        extra_env: Optional[Dict[str, str]] = None,
        command_builder: Optional[Callable[[DeploySpec], List[str]]] = None,
        format_checker: Optional[Callable[[str], str]] = None,
        autoload: bool = True,
    ) -> None:
        self.deploys_dir = Path(deploys_dir)
        self.python = python
        self.device = device
        self.poll_interval = max(0.1, float(poll_interval))
        self.max_log_lines = max(100, int(max_log_lines))
        self.extra_env: Dict[str, str] = dict(extra_env or {})
        self.command_builder = command_builder
        # 判断某格式在本机是否可用（返回空串=可用，否则是原因）。
        # 可在测试里覆盖，以便用假后端验证多格式流程而不受本机安装情况影响。
        self.format_checker: Callable[[str], str] = format_checker or unavailable

        self._lock = threading.RLock()
        self._runtimes: Dict[str, DeployRuntime] = {}
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._on_finish: List[Callable[[DeployJob, Optional[DeployResult]], None]] = []
        self._resources = ResourceMonitor()
        self._stop_event = threading.Event()
        self._monitor: Optional[threading.Thread] = None

        self.deploys_dir.mkdir(parents=True, exist_ok=True)
        if autoload:
            self._load_existing()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="deploy-monitor", daemon=True
        )
        self._monitor.start()

    # =================================================================
    # 查询
    # =================================================================

    def list(self) -> List[DeployJob]:
        self.refresh()
        with self._lock:
            jobs = [rt.job for rt in self._runtimes.values()]
        return sorted(jobs, key=lambda j: (j.created_at, j.id), reverse=True)

    def get(self, deploy_id: str) -> DeployJob:
        with self._lock:
            rt = self._runtimes.get(deploy_id)
        if rt is None:
            self.refresh()
            with self._lock:
                rt = self._runtimes.get(deploy_id)
        if rt is None:
            raise DeployJobNotFound(f"导出任务不存在: {deploy_id}")
        return rt.job

    def refresh(self) -> int:
        with self._lock:
            known = set(self._runtimes)
        added = 0
        for job in load_jobs(self.deploys_dir):
            if job.id in known:
                continue
            try:
                self._load_one(job)
                added += 1
            except Exception:
                continue
        return added

    def _runtime(self, deploy_id: str) -> DeployRuntime:
        with self._lock:
            rt = self._runtimes.get(deploy_id)
        if rt is None:
            raise DeployJobNotFound(f"导出任务不存在: {deploy_id}")
        return rt

    def result(self, deploy_id: str) -> Optional[DeployResult]:
        rt = self._runtime(deploy_id)
        if rt.result is None:
            rt.result = load_result(rt.run_dir)
        return rt.result

    def verify(self, deploy_id: str):
        """重新校验产物完整性（M4-03）。"""
        result = self.result(deploy_id)
        if result is None:
            raise DeployStateError("导出尚无结果，无法校验")
        return verify_result(result)

    def logs(self, deploy_id: str, offset: int = 0, limit: int = 500) -> Dict[str, Any]:
        rt = self._runtime(deploy_id)
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        lines = [{"seq": seq, "text": text} for seq, text in rt.logs if seq >= offset][:limit]
        return {"lines": lines, "total": rt.log_total, "next_offset": offset + len(lines)}

    def resources(self, deploy_id: Optional[str] = None) -> Dict[str, Any]:
        if deploy_id is None:
            return self._resources.sample(None)
        rt = self._runtime(deploy_id)
        if rt.resources:
            return rt.resources
        return self._resources.sample(rt.job.pid or rt.adopted_pid)

    def capabilities(self) -> Dict[str, Any]:
        return {"resources": self._resources.capabilities(), "formats": capability_report()}

    # =================================================================
    # 事件
    # =================================================================

    def subscribe(self, deploy_id: str, callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.setdefault(deploy_id, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                subs = self._subscribers.get(deploy_id)
                if subs and callback in subs:
                    subs.remove(callback)
                if subs is not None and not subs:
                    self._subscribers.pop(deploy_id, None)

        return unsubscribe

    def on_finish(
        self, callback: Callable[[DeployJob, Optional[DeployResult]], None]
    ) -> Callable[[], None]:
        with self._lock:
            if callback in self._on_finish:
                return lambda: None
            self._on_finish.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._on_finish:
                    self._on_finish.remove(callback)

        return unsubscribe

    def _emit(self, deploy_id: str, event: Dict[str, Any]) -> None:
        event.setdefault("job_id", deploy_id)
        with self._lock:
            subs = list(self._subscribers.get(deploy_id, []))
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass

    def _notify_finish(self, job: DeployJob, result: Optional[DeployResult]) -> None:
        with self._lock:
            callbacks = list(self._on_finish)
        for cb in callbacks:
            try:
                cb(job, result)
            except Exception:
                pass

    def snapshot(self, deploy_id: str, log_limit: int = 200) -> Dict[str, Any]:
        rt = self._runtime(deploy_id)
        logs = self.logs(deploy_id, offset=max(0, rt.log_total - log_limit), limit=log_limit)
        result = self.result(deploy_id)
        return {
            "type": "snapshot",
            "job_id": deploy_id,
            "job": rt.job.to_dict(),
            "result": result.to_dict() if result else None,
            "logs": logs,
            "resources": rt.resources,
            "monitor": self._resources.capabilities(),
        }

    # =================================================================
    # 启动 / 停止
    # =================================================================

    def start(self, spec: DeploySpec) -> DeployJob:
        self._apply_defaults(spec)
        problems = spec.validate()
        if problems:
            raise ValueError("；".join(problems))

        # 不支持的格式提前拒绝，避免启动一个注定失败的子进程
        unavailable_formats = {
            fmt: self.format_checker(fmt)
            for fmt in spec.normalized_formats()
            if self.format_checker(fmt)
        }
        if unavailable_formats:
            detail = "；".join(f"{k}({v})" for k, v in unavailable_formats.items())
            raise ValueError(f"以下格式在本机不可用: {detail}")

        run_dir = spec.run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"导出目录非空: {run_dir}")

        job = DeployJob(
            id=spec.name,
            status=STATUS_PENDING,
            spec=spec.to_dict(),
            run_dir=str(run_dir),
            created_at=now_iso(),
        )
        rt = DeployRuntime(job=job, spec=spec, finalize_done=threading.Event())
        with self._lock:
            self._runtimes[job.id] = rt

        self._spawn(rt)
        return job

    def stop(self, deploy_id: str) -> DeployJob:
        rt = self._runtime(deploy_id)
        if not is_active(rt.job.status):
            raise DeployStateError(f"导出任务当前状态为 {rt.job.status}，无法停止")

        rt.job.stopped_by_user = True
        rt.job.status = STATUS_STOPPING
        rt.job.message = "正在停止导出进程"
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

    def _apply_defaults(self, spec: DeploySpec) -> None:
        if not spec.python:
            spec.python = self.python
        if not spec.device or spec.device == "auto":
            spec.device = self.device
        if not spec.name:
            spec.name = make_deploy_id(spec.tag)
        if not spec.project:
            spec.project = str(self.deploys_dir)
        spec.project = str(Path(spec.project).expanduser().resolve())

    def _build_command(self, spec: DeploySpec) -> List[str]:
        if self.command_builder is not None:
            return list(self.command_builder(spec))
        return [
            spec.python or sys.executable,
            "-m",
            "core.deploy.runner",
            "--spec",
            str(spec.spec_path),
        ]

    def _spawn(self, rt: DeployRuntime) -> None:
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
            rt.job.error = f"启动导出进程失败: {exc}"
            rt.job.ended_at = now_iso()
            self._persist(rt)
            self._emit_status(rt)
            return

        rt.job.pid = rt.process.pid
        rt.job.status = STATUS_RUNNING
        self._open_log_file(rt)
        self._append_log(rt, f"[studio] 启动导出进程 pid={rt.process.pid}")
        self._append_log(rt, f"[studio] 命令: {' '.join(command)}")

        self._persist(rt)
        self._emit_status(rt)

        reader = threading.Thread(target=self._pump_logs, args=(rt,), daemon=True)
        reader.start()

    def _open_log_file(self, rt: DeployRuntime) -> None:
        if rt.log_handle is not None:
            return
        try:
            rt.log_handle = open(rt.run_dir / "deploy.log", "a", encoding="utf-8")
        except Exception:
            rt.log_handle = None

    def _append_log(self, rt: DeployRuntime, text: str) -> None:
        rt.add_log(text, self.max_log_lines)
        if rt.log_handle is not None:
            try:
                rt.log_handle.write(text + "\n")
                rt.log_handle.flush()
            except Exception:
                rt.log_handle = None

    def _pump_logs(self, rt: DeployRuntime) -> None:
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

    def _tick(self, rt: DeployRuntime) -> None:
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

    def _flush_logs(self, rt: DeployRuntime) -> None:
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

    def _finalize(self, rt: DeployRuntime, returncode: Optional[int]) -> None:
        if rt.finalized:
            # 已有人在收尾：等它真正跑完再返回，否则调用方会看到中间状态
            if rt.finalize_done is not None:
                rt.finalize_done.wait(timeout=10.0)
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

        if rt.job.stopped_by_user:
            status, message = STATUS_STOPPED, "已手动停止"
        elif result is not None and result.ok:
            status, message = STATUS_FINISHED, "导出完成"
        elif result is not None:
            status, message = STATUS_FAILED, "导出失败"
            rt.job.error = result.error or "所有格式导出失败"
        elif returncode == 0:
            status, message = STATUS_FINISHED, "导出完成"
        elif returncode is None:
            status, message = STATUS_INTERRUPTED, "导出进程已结束，未产出结果"
        else:
            status, message = STATUS_FAILED, "导出失败"
            rt.job.error = f"导出进程退出码 {returncode}"

        if result is not None:
            rt.job.artifacts = [a.to_dict() for a in result.artifacts]

        self._flush_logs(rt)
        rt.job.status = status
        rt.job.message = message
        self._persist(rt)
        self._close_log_file(rt)
        # 先让上层把产物挂到模型库，再对外宣告 finished
        self._notify_finish(rt.job, result)
        self._emit_status(rt)
        self._emit(
            rt.job.id,
            {
                "type": "finished",
                "status": rt.job.status,
                "returncode": returncode,
                "result": result.to_dict() if result else None,
            },
        )
        # 收尾彻底完成，唤醒可能正在等它的 stop()
        if rt.finalize_done is not None:
            rt.finalize_done.set()

    def _emit_status(self, rt: DeployRuntime) -> None:
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

    def _persist(self, rt: DeployRuntime) -> None:
        try:
            save_job(rt.job, rt.run_dir)
        except Exception:
            pass

    @staticmethod
    def _close_log_file(rt: DeployRuntime) -> None:
        if rt.log_handle is not None:
            try:
                rt.log_handle.close()
            except Exception:
                pass
            rt.log_handle = None

    def _load_existing(self) -> None:
        for job in load_jobs(self.deploys_dir):
            try:
                self._load_one(job)
            except Exception:
                continue

    def _load_one(self, job: DeployJob) -> None:
        spec = DeploySpec.from_dict(job.spec or {})
        rt = DeployRuntime(job=job, spec=spec, finalize_done=threading.Event())
        self._load_log_history(rt)

        if is_active(job.status):
            if job.pid and pid_alive(job.pid):
                rt.adopted_pid = job.pid
                job.status = STATUS_RUNNING
                job.message = (
                    f"服务重启后接管 pid={job.pid}；结果将在进程结束后从 deploy_result.json 读取"
                )
            else:
                job.status = STATUS_INTERRUPTED
                job.message = "服务重启时导出进程已不存在"
                job.error = job.error or "进程失联"
                job.ended_at = job.ended_at or now_iso()
                job.pid = None
            save_job(job, rt.run_dir)

        rt.result = load_result(rt.run_dir)
        if rt.result is not None and not job.artifacts:
            job.artifacts = [a.to_dict() for a in rt.result.artifacts]
        if is_terminal(job.status):
            rt.finalized = True

        with self._lock:
            self._runtimes[job.id] = rt

    def _load_log_history(self, rt: DeployRuntime) -> None:
        log_path = rt.run_dir / "deploy.log"
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


def _write_spec(spec: DeploySpec) -> Path:
    path = spec.spec_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
