"""数据集划分模块。"""

from __future__ import annotations

from .splitter import (
    STRATEGY_RANDOM,
    STRATEGY_STRATIFIED,
    UNLABELED,
    SplitConfig,
    SplitReport,
    assign_splits,
)

__all__ = [
    "STRATEGY_RANDOM",
    "STRATEGY_STRATIFIED",
    "UNLABELED",
    "SplitConfig",
    "SplitReport",
    "assign_splits",
]
