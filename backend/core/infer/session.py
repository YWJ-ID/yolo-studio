"""推理会话（主进程侧）。

管理常驻推理子进程的生命周期，并用**请求-响应配对**把「发出的帧」
和「回来的结果」对上：

    session = InferSession()
    session.load(spec)                 # 等 loaded 回执
    result = session.infer_jpeg(b64)   # 阻塞等待本帧结果，带超时

要点：
- 单进程单会话（一个服务同时只跑一个模型）。要换模型就重新 load。
- 读线程把子进程每一行 JSON 解析后按类型分发：`loaded` 唤醒 load 的等待者，
  `result` 按 id 唤醒对应帧的等待者。
- 超时不重试，直接报错；调用方可选择重启会话（见 `restart_on_timeout`）。
- 子进程崩溃时，所有等待者被唤醒并收到明确的错误，不会无限挂住。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..process import kill_tree
from .result import FrameResult, build_frame_result
from .spec import InferOptions, InferSpec

BACKEND_DIR = Path(__file__).resolve().parents[2]

DEFAULT_LOAD_TIMEOUT = 120.0   # 首次加载要 import torch，给足时间
DEFAULT_FRAME_TIMEOUT = 30.0   # 单帧推理超时


class InferError(RuntimeError):
    """推理会话层面的错误（加载失败 / 超时 / 子进程异常）。"""


class InferSession:
    """一个常驻推理子进程的封装。非线程安全：同一实例的调用需自行串行化。"""

    def __init__(self, python: Optional[str] = None) -> None:
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self.info: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        # 已加载完成的信息（load 的返回）
        self._loaded: Optional[Dict[str, Any]] = None
        # 帧结果：id -> payload
        self._results: Dict[int, Dict[str, Any]] = {}
        # 全局错误（子进程 stderr / 崩溃）
        self.last_error: str = ""
        self._reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self._closing = False

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """启动子进程与读线程（幂等：已在运行则不重复启动）。"""
        if self.proc is not None and self.proc.poll() is None:
            return

        self.proc = subprocess.Popen(
            [self.python, "-m", "core.infer.worker"],
            cwd=str(BACKEND_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._closing = False
        self._loaded = None
        self._results.clear()
        self.last_error = ""
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def stop(self, timeout: float = 8.0) -> None:
        """优雅关闭子进程。"""
        proc = self.proc
        if proc is None:
            return
        self._closing = True
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except Exception:
            kill_tree(proc.pid)
        self.proc = None
        # 唤醒所有等待者
        with self._cv:
            self._cv.notify_all()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _check_dead(self) -> None:
        """在等待循环里主动探测子进程是否已死。

        Windows 上读线程的 `for line in stdout` 不保证在子进程退出时立刻返回，
        因此不能只依赖读线程唤醒等待者——等待循环本身也要看 `poll()`。
        """
        proc = self.proc
        if proc is not None and proc.poll() is not None and not self.last_error:
            self.last_error = f"推理子进程已退出（退出码 {proc.poll()}）"

    # ---------- 读线程 ----------

    def _read_stdout(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._dispatch(msg)
        # 子进程结束：唤醒所有等待者
        with self._cv:
            self._closing = True
            if not self.last_error:
                code = proc.poll()
                self.last_error = f"推理子进程已退出（退出码 {code}）"
            self._cv.notify_all()

    def _read_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.last_error = line

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        with self._cv:
            if mtype == "loaded":
                self._loaded = msg
                self._cv.notify_all()
            elif mtype == "result":
                fid = msg.get("id")
                if fid is not None:
                    self._results[int(fid)] = msg
                self._cv.notify_all()
            elif mtype == "error":
                self.last_error = str(msg.get("error") or "未知错误")
                self._cv.notify_all()

    # ---------- 命令 ----------

    def _write(self, obj: Dict[str, Any]) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise InferError(self.last_error or "推理子进程未运行")
        try:
            proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise InferError(f"向推理子进程写入失败: {exc}") from exc

    def load(self, spec: InferSpec, timeout: float = DEFAULT_LOAD_TIMEOUT) -> Dict[str, Any]:
        """加载权重。成功返回描述信息，失败抛 InferError。"""
        problems = spec.validate()
        if problems:
            raise InferError("; ".join(problems))

        self.start()
        with self._cv:
            self._loaded = None
            self.last_error = ""

        self._write({"type": "load", "spec": spec.to_dict()})

        deadline = time.time() + timeout
        with self._cv:
            while self._loaded is None:
                self._check_dead()
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise InferError(f"加载权重超时（{timeout:.0f}s）")
                if not self.alive and self._loaded is None:
                    raise InferError(self.last_error or "推理子进程已退出")
                self._cv.wait(timeout=min(remaining, 0.5))
            loaded = self._loaded

        if not loaded.get("ok"):
            raise InferError(str(loaded.get("error") or "加载失败"))
        self.info = loaded
        return loaded

    def infer_jpeg(
        self,
        jpeg_b64: str,
        options: Optional[InferOptions] = None,
        frame_id: Optional[int] = None,
        timeout: float = DEFAULT_FRAME_TIMEOUT,
    ) -> FrameResult:
        """对一帧（base64 JPEG）推理，阻塞等待结果。"""
        options = options or InferOptions()
        if frame_id is None:
            frame_id = int(time.time() * 1000) % 1000000

        self._write({"type": "frame", "id": frame_id, "jpeg": jpeg_b64,
                     "options": options.to_dict()})

        deadline = time.time() + timeout
        with self._cv:
            while frame_id not in self._results:
                self._check_dead()
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise InferError(f"推理超时（{timeout:.0f}s）")
                if not self.alive:
                    raise InferError(self.last_error or "推理子进程已退出")
                self._cv.wait(timeout=min(remaining, 0.5))
            msg = self._results.pop(frame_id)

        duration = msg.get("duration_ms")
        return build_frame_result(msg, frame_id=frame_id, duration_ms=duration)

    def infer_path(
        self,
        path: str,
        options: Optional[InferOptions] = None,
        timeout: float = DEFAULT_FRAME_TIMEOUT,
    ) -> FrameResult:
        """直接对服务器上的图片文件推理（不经过 base64，省一次编解码）。"""
        options = options or InferOptions()
        frame_id = int(time.time() * 1000) % 1000000
        self._write({"type": "frame", "id": frame_id, "path": str(path),
                     "options": options.to_dict()})

        deadline = time.time() + timeout
        with self._cv:
            while frame_id not in self._results:
                self._check_dead()
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise InferError(f"推理超时（{timeout:.0f}s）")
                if not self.alive:
                    raise InferError(self.last_error or "推理子进程已退出")
                self._cv.wait(timeout=min(remaining, 0.5))
            msg = self._results.pop(frame_id)

        return build_frame_result(msg, frame_id=frame_id, duration_ms=msg.get("duration_ms"))

    # ---------- 便捷属性 ----------

    @property
    def task(self) -> str:
        return str(self.info.get("task") or "")

    @property
    def classes(self) -> List[str]:
        return list(self.info.get("classes") or [])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "alive": self.alive,
            "loaded": bool(self.info),
            "task": self.task,
            "classes": self.classes,
            "num_classes": len(self.classes),
            "class_source": self.info.get("class_source", ""),
            "weights": self.info.get("weights", ""),
            "format": self.info.get("format", ""),
            "device": self.info.get("device", ""),
            "imgsz": self.info.get("imgsz", 0),
            "last_error": self.last_error,
        }


__all__ = ["DEFAULT_FRAME_TIMEOUT", "DEFAULT_LOAD_TIMEOUT", "InferError", "InferSession"]
