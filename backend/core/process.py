"""进程工具：判断存活、终止整棵进程树。

训练与评估都要用，因此放在公共位置，避免两处各写一份（容易只改一处）。

为什么不能只用 `Popen.terminate()`：ultralytics 会按 `workers` 拉起 dataloader
子进程，只杀父进程会留下孤儿进程占着 CPU 和内存。
"""

from __future__ import annotations

import os
import signal
from typing import List


def pid_alive(pid: int) -> bool:
    """进程是否存在。优先用 psutil（能识别僵尸状态），否则退化为 os.kill(pid, 0)。"""
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False


def kill_tree(pid: int, timeout: float = 8.0) -> List[int]:
    """终止整个进程树，返回未能确认终止的 pid 列表（通常为空）。

    先 terminate 给进程收尾机会，超时后再 kill。
    """
    if not pid:
        return []
    killed: List[int] = []
    try:
        import psutil

        parent = psutil.Process(int(pid))
        procs = parent.children(recursive=True) + [parent]
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
        _, alive = psutil.wait_procs(procs, timeout=timeout)
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass
        return [p.pid for p in alive]
    except Exception:
        # 没有 psutil 或进程已消失
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            return []
        return []


__all__ = ["kill_tree", "pid_alive"]
