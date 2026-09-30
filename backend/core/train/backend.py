"""训练后端抽象。

一期只实现 ultralytics，但调度、日志、指标推送等上层逻辑只依赖这个接口，
将来接 YOLOv5 / MMDetection 时只需新增一个实现。

接口刻意保持三个纯函数式的钩子：
    resolve_weights()  把权重写法解析成实际可用路径
    build_command()    组装子进程命令行
    parse_metrics()    从训练目录读取指标

这些都是无副作用的，便于单独测试——不需要真的启动训练。
"""

from __future__ import annotations

import abc
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import RESULTS_CSV, MetricsSeries, parse_results_csv
from .spec import TrainSpec


class TrainerBackend(abc.ABC):
    """训练框架适配器。"""

    name: str = "base"
    display_name: str = "基础后端"

    # ---------- 必须实现 ----------

    @abc.abstractmethod
    def resolve_weights(self, weights: str, base_dir: Optional[Path] = None) -> str:
        """把权重写法解析成实际可用的路径或名称。"""

    @abc.abstractmethod
    def build_command(self, spec: TrainSpec) -> List[str]:
        """组装启动训练的完整命令行（不含 shell 解析）。"""

    def parse_metrics(self, csv_path) -> MetricsSeries:
        """解析指标文件。默认实现读 results.csv。"""
        return parse_results_csv(csv_path)

    # ---------- 产物位置（各后端可能有差异） ----------

    def results_csv(self, run_dir) -> Path:
        return Path(run_dir) / RESULTS_CSV

    def weights_dir(self, run_dir) -> Path:
        return Path(run_dir) / "weights"

    def best_weights(self, run_dir) -> Optional[Path]:
        p = self.weights_dir(run_dir) / "best.pt"
        return p if p.is_file() else None

    def last_weights(self, run_dir) -> Optional[Path]:
        p = self.weights_dir(run_dir) / "last.pt"
        return p if p.is_file() else None

    # ---------- 通用工具 ----------

    def metric_columns(self) -> List[str]:
        """该后端产出的核心指标名，供前端选图。"""
        return ["mAP50", "mAP50-95", "precision", "recall", "train_loss"]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "metrics": self.metric_columns(),
        }


def interpreter(spec: TrainSpec) -> str:
    """训练子进程使用的解释器：spec 未指定时用当前解释器。"""
    return spec.python or sys.executable
