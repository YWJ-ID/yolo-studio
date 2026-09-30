"""训练指标解析。

指标来源是 ultralytics 写在训练目录下的 `results.csv`，**不 hook 其内部**：
升级 ultralytics 只需要这个解析器跟着列名变化调整，可视化逻辑不受影响。

results.csv 由 ultralytics 逐轮追加，首行是列名，例如检测任务：

    epoch,time,train/box_loss,...,metrics/precision(B),metrics/recall(B),
    metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,...,lr/pg0,...

分类任务则是 `train/loss` / `metrics/accuracy_top1` / `metrics/accuracy_top5` 等。
解析器不写死列名，只按列名取值，因此两种任务都能用。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 训练目录下的指标文件名
RESULTS_CSV = "results.csv"

# 对外披露的「重点指标」：raw 列名 -> 展示名。
# 找不到的列直接跳过，不补 0，避免把「没有这个指标」显示成 0。
HEADLINE_METRICS = (
    ("metrics/mAP50(B)", "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50-95"),
    ("metrics/precision(B)", "precision"),
    ("metrics/recall(B)", "recall"),
    ("metrics/accuracy_top1", "top1"),
    ("metrics/accuracy_top5", "top5"),
)

# 损失列，按任务不同取不同组合
DETECTION_LOSSES = ("train/box_loss", "train/cls_loss", "train/dfl_loss")


@dataclass
class MetricRow:
    """一轮的指标。"""

    epoch: int
    values: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str) -> Optional[float]:
        return self.values.get(key)

    def headline(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for raw, label in HEADLINE_METRICS:
            v = self.values.get(raw)
            if v is not None:
                out[label] = v
        loss = self.total_loss()
        if loss is not None:
            out["train_loss"] = loss
        return out

    def total_loss(self) -> Optional[float]:
        """训练损失合计。检测任务把三个分项相加，分类任务直接取 train/loss。"""
        if "train/loss" in self.values:
            return self.values["train/loss"]
        parts = [self.values.get(k) for k in DETECTION_LOSSES]
        present = [p for p in parts if p is not None]
        if not present:
            return None
        return round(float(sum(present)), 6)

    def to_dict(self) -> Dict[str, Any]:
        return {"epoch": self.epoch, "values": self.values, "headline": self.headline()}


@dataclass
class MetricsSeries:
    """按轮次排列的指标序列。"""

    columns: List[str] = field(default_factory=list)
    rows: List[MetricRow] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.rows)

    @property
    def epochs(self) -> int:
        """已完成的轮数（不是最大 epoch 值，断点续训后两者可能不同）。"""
        return len(self.rows)

    @property
    def last_epoch(self) -> Optional[int]:
        return self.rows[-1].epoch if self.rows else None

    def latest(self) -> Optional[MetricRow]:
        return self.rows[-1] if self.rows else None

    def best(self, key: str, mode: str = "max") -> Optional[MetricRow]:
        """按某个指标取最优轮次。找不到该指标时返回 None。"""
        candidates = [r for r in self.rows if r.get(key) is not None]
        if not candidates:
            return None
        pick = max if mode == "max" else min
        return pick(candidates, key=lambda r: r.get(key))  # type: ignore[arg-type]

    def best_headline(self) -> Dict[str, Any]:
        """择优展示：mAP 取最大，损失取最小。"""
        out: Dict[str, Any] = {}
        m = self.best("metrics/mAP50-95(B)") or self.best("metrics/mAP50(B)")
        if m:
            out["best_epoch"] = m.epoch
            out["best"] = m.headline()
        return out

    def to_dict(self, max_rows: int = 0) -> Dict[str, Any]:
        rows = self.rows if max_rows <= 0 else self.rows[-max_rows:]
        return {
            "columns": self.columns,
            "epochs": self.epochs,
            "last_epoch": self.last_epoch,
            "rows": [r.to_dict() for r in rows],
            "best": self.best_headline(),
        }

    def progress(self, total_epochs: int) -> Dict[str, Any]:
        """训练进度。total_epochs 来自 spec，用于算百分比。"""
        done = self.epochs
        total = max(1, int(total_epochs or 0))
        return {
            "epoch": self.last_epoch if self.last_epoch is not None else done,
            "epochs_done": done,
            "epochs_total": total,
            "percent": round(min(1.0, done / total), 4),
        }


def _to_float(text: str) -> Optional[float]:
    """把单元格转成 float。ultralytics 偶尔写出空串或 nan。"""
    s = (text or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v:  # NaN
        return None
    return v


def parse_results_csv(path, since_epoch: Optional[int] = None) -> MetricsSeries:
    """解析 results.csv。

    since_epoch 给定时只返回 epoch 大于它的行，用于增量推送。

    容错处理：
      * 文件不存在 / 为空 -> 空序列（训练刚开始时是常态，不是错误）；
      * 最后一行可能只写了一半（训练进程正在写）-> 列数不足则丢弃；
      * 同一 epoch 出现多次（断点续训会重写）-> 保留最后一次；
      * epoch 不是整数 -> 跳过。
    """
    p = Path(path)
    series = MetricsSeries()
    if not p.is_file():
        return series

    try:
        text = p.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return series
    if not text.strip():
        return series

    lines = text.splitlines()
    if not lines:
        return series

    reader = csv.reader(lines)
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        return series

    if "epoch" not in header:
        return series
    epoch_idx = header.index("epoch")
    series.columns = header

    # epoch -> 行，用于处理重复 epoch
    merged: Dict[int, MetricRow] = {}
    for raw in reader:
        if len(raw) < len(header):
            continue  # 半行（正在写入）或损坏
        epoch_text = raw[epoch_idx].strip()
        try:
            epoch = int(float(epoch_text))
        except (ValueError, TypeError):
            continue

        values: Dict[str, float] = {}
        for i, name in enumerate(header):
            if i == epoch_idx:
                continue
            v = _to_float(raw[i])
            if v is not None:
                values[name] = v
        merged[epoch] = MetricRow(epoch=epoch, values=values)

    rows = [merged[e] for e in sorted(merged)]
    if since_epoch is not None:
        rows = [r for r in rows if r.epoch > since_epoch]
    series.rows = rows
    return series
