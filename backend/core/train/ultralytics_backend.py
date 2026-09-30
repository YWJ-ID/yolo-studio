"""ultralytics 训练后端。

关键约束：
  * 训练跑在**独立子进程**里（见 `runner.py`），本模块不在 API 进程中导入 ultralytics；
  * `build_command()` 只组装命令行，不导入任何重依赖，因此可以在 API 进程里安全调用；
  * `train()` 才真正 import ultralytics，且只在训练子进程里执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import TrainerBackend, interpreter
from .metrics import MetricsSeries, parse_results_csv
from .spec import DEFAULT_WEIGHTS, TrainSpec


class UltralyticsBackend(TrainerBackend):
    """调用 ultralytics 的 YOLO.train()。"""

    name = "ultralytics"
    display_name = "Ultralytics YOLO"

    # ---------- 权重解析 ----------

    def resolve_weights(self, weights: str, base_dir: Optional[Path] = None) -> str:
        w = (weights or "").strip()
        if not w:
            raise ValueError("未指定权重（可用 yolo11n.yaml 从零训练，或用 .pt 预训练权重）")

        # 1) 已经是存在的文件路径
        p = Path(w)
        if p.is_file():
            return str(p.resolve())

        # 2) 结构文件：ultralytics 包内自带，无需存在性检查
        if w.lower().endswith(".yaml"):
            return w

        # 3) 在权重目录里按文件名找
        if base_dir:
            for cand in (Path(base_dir) / w, Path(base_dir) / "weights" / w):
                if cand.is_file():
                    return str(cand.resolve())

        # 4) 交给 ultralytics 处理（首次使用会自动下载）
        return w

    def default_weights(self, task: str) -> str:
        return DEFAULT_WEIGHTS.get(task, DEFAULT_WEIGHTS["detect"])

    # ---------- 命令行 ----------

    def build_command(self, spec: TrainSpec) -> List[str]:
        """启动训练子进程的命令行。

        实际训练参数通过 `run_dir/train_spec.json` 传递，命令行只负责指向它。
        这样参数再多也不会撞上 Windows 的命令行长度限制。
        """
        return [
            interpreter(spec),
            "-m",
            "core.train.runner",
            "--spec",
            str(spec.spec_path),
        ]

    # ---------- 训练参数 ----------

    def train_kwargs(self, spec: TrainSpec, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """转换成 YOLO.train() 的关键字参数。纯函数，可单独测试。"""
        weights = self.resolve_weights(spec.weights or self.default_weights(spec.task), base_dir)

        kwargs: Dict[str, Any] = {
            "data": spec.data_yaml,
            "epochs": spec.epochs,
            "imgsz": spec.imgsz,
            "batch": spec.batch,
            "device": spec.device,
            "workers": spec.workers,
            "project": spec.project,
            "name": spec.name,
            "exist_ok": True,       # 目录由 manager 预先创建，避免被加后缀
            "seed": spec.seed,
            "patience": spec.patience,
            "optimizer": spec.optimizer,
            "task": spec.task,
            "verbose": True,
        }
        if spec.lr0 is not None:
            kwargs["lr0"] = spec.lr0
        if spec.resume:
            # 断点续训时 ultralytics 会忽略除 resume 外的多数参数，取自 checkpoint
            kwargs["resume"] = True
        kwargs.update(spec.extra)
        return {"weights": weights, "kwargs": kwargs}

    def parse_metrics(self, csv_path) -> MetricsSeries:
        return parse_results_csv(csv_path)

    def metric_columns(self) -> List[str]:
        return ["mAP50", "mAP50-95", "precision", "recall", "train_loss"]


def train(spec: TrainSpec) -> int:
    """在训练子进程中执行 ultralytics 训练。返回进程退出码。

    这是唯一 import ultralytics 的地方，只应被 runner 调用。
    """
    from ultralytics import YOLO  # 延迟导入：只发生在训练子进程里

    backend = UltralyticsBackend()
    prepared = backend.train_kwargs(spec, base_dir=Path(spec.project))
    weights = prepared["weights"]
    kwargs = prepared["kwargs"]

    model = YOLO(weights)
    model.train(**kwargs)
    return 0


__all__ = ["UltralyticsBackend", "train"]
