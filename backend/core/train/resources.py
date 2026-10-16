"""资源监控：CPU（psutil）+ GPU（pynvml，可选）。

降级要求：本机没有 NVIDIA 卡，pynvml 也可能没安装。
两种情况都必须返回结构完整的 CPU 数据，并把 GPU 标记为不可用，
让前端统一处理，而不是靠 try/except 猜。

注意：这里的"GPU 不可用"和 CUDA 是否可用是两件事——
开了 CUDA 的机器也可能没装 pynvml，此时训练在跑 GPU 但监控看不到利用率。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class ResourceMonitor:
    """周期性采样训练进程与整机的资源占用。"""

    def __init__(self) -> None:
        self._psutil = None
        self._nvml = None
        self._gpu_handles: list = []
        self._gpu_error = ""
        self._procs: Dict[int, Any] = {}
        self._init_psutil()
        self._init_nvml()

    # ---------- 初始化 ----------

    def _init_psutil(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            # 第一次调用只是建立基线，否则首帧 cpu_percent 会返回 0
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                self._gpu_error = "未检测到 NVIDIA 设备"
                return
            self._nvml = pynvml
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception as exc:
            self._gpu_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    # ---------- 能力探测 ----------

    def capabilities(self) -> Dict[str, Any]:
        return {
            "cpu_monitor": self._psutil is not None,
            "gpu_monitor": self._nvml is not None,
            "gpu_count": len(self._gpu_handles),
            "gpu_error": self._gpu_error,
        }

    # ---------- 采样 ----------

    def cpu(self, pid: Optional[int] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": self._psutil is not None}
        if not self._psutil:
            return out

        psutil = self._psutil
        try:
            out["percent"] = psutil.cpu_percent(interval=None)
            out["count"] = psutil.cpu_count() or 0
            vm = psutil.virtual_memory()
            out["mem_percent"] = vm.percent
            out["mem_used"] = vm.used
            out["mem_total"] = vm.total
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"

        proc_info = self._process(pid)
        if proc_info:
            out["process"] = proc_info
        return out

    def _process(self, pid: Optional[int]) -> Optional[Dict[str, Any]]:
        """采样训练进程**整棵树**的占用。

        为什么要走进程树：Windows 上 venv 的 `python.exe` 是个外壳，会把真正的
        解释器作为**子进程**启动（实测外壳只有几 MB、1 个线程），只采 `pid` 一个进程
        会严重低估 CPU / 内存。开启 `workers>0` 时还会再多出 dataloader 子进程，
        同样属于这次训练的开销。
        """
        if not self._psutil or not pid:
            return None
        psutil = self._psutil

        root = self._get_proc(pid)
        if root is None:
            return None
        try:
            if not root.is_running():
                return None
        except Exception:
            return None

        # 收集整棵树的 pid（根 + 递归子进程）
        pids = [int(pid)]
        try:
            pids += [c.pid for c in root.children(recursive=True)]
        except Exception:
            pass

        cpu = 0.0
        mem_rss = 0
        threads = 0
        live = 0
        for p in pids:
            proc = self._get_proc(p)
            if proc is None:
                continue
            try:
                if not proc.is_running():
                    continue
            except Exception:
                continue
            live += 1
            try:
                cpu += proc.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                mem_rss += proc.memory_info().rss
            except Exception:
                pass
            try:
                threads += proc.num_threads()
            except Exception:
                pass

        if live == 0:
            return None
        try:
            mem_total = psutil.virtual_memory().total or 0
        except Exception:
            mem_total = 0
        return {
            "pid": int(pid),
            "processes": live,  # 进程树内的进程数（含外壳）
            "cpu_percent": round(cpu, 1),
            "mem_rss": mem_rss,
            "mem_percent": round(mem_rss / mem_total * 100, 2) if mem_total else 0.0,
            "threads": threads,
            "children": max(0, live - 1),
        }

    def _get_proc(self, pid: int):
        """按 pid 取缓存的 psutil.Process（缓存是为了让 cpu_percent 有时间基线）。"""
        proc = self._procs.get(pid)
        if proc is not None and proc.pid == pid:
            return proc
        try:
            proc = self._psutil.Process(pid)
        except Exception:
            return None
        self._procs[pid] = proc
        try:
            proc.cpu_percent(interval=None)  # 建立基线，首帧返回 0
        except Exception:
            pass
        return proc

    def gpu(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "available": self._nvml is not None,
            "devices": [],
            "error": self._gpu_error,
        }
        if not self._nvml:
            return out

        pynvml = self._nvml
        for i, handle in enumerate(self._gpu_handles):
            dev: Dict[str, Any] = {"index": i}
            try:
                dev["name"] = pynvml.nvmlDeviceGetName(handle)
                if isinstance(dev["name"], bytes):
                    dev["name"] = dev["name"].decode("utf-8", "replace")
            except Exception:
                dev["name"] = f"GPU {i}"
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                dev["utilization"] = util.gpu
                dev["memory_utilization"] = util.memory
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                dev["mem_used"] = mem.used
                dev["mem_total"] = mem.total
                dev["mem_percent"] = round(mem.used / mem.total * 100, 1) if mem.total else 0.0
            except Exception:
                pass
            try:
                dev["temperature"] = pynvml.nvmlDeviceGetTemperature(handle, 0)
            except Exception:
                pass
            out["devices"].append(dev)
        return out

    def sample(self, pid: Optional[int] = None) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "cpu": self.cpu(pid),
            "gpu": self.gpu(),
        }

    def close(self) -> None:
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
        self._gpu_handles = []
        self._procs.clear()


def resolve_device(configured: str = "") -> str:
    """确定训练设备。

    `auto`（或留空）时才去探测 CUDA；torch 是重依赖，import 只发生在这一条路径上，
    显式指定设备（如 cpu / cuda:0）时不会加载它。
    """
    c = (configured or "").strip()
    if c and c.lower() != "auto":
        return c
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0" if torch.cuda.device_count() else "cpu"
    except Exception:
        pass
    return "cpu"


_default: Optional[ResourceMonitor] = None


def default_monitor() -> ResourceMonitor:
    global _default
    if _default is None:
        _default = ResourceMonitor()
    return _default
