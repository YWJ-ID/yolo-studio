"""数据清洗模块。"""

from __future__ import annotations

from .engine import clean, list_rules
from .findings import (
    ACTION_CLAMP,
    ACTION_REMOVE_ANNOTATION,
    ACTION_REMOVE_IMAGE,
    ACTION_REPORT,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_ORDER,
    SEVERITY_WARNING,
    CleanReport,
    Finding,
)
from .rules import RULES, RULES_BY_ID, CleanConfig, CleanContext, Rule

__all__ = [
    "ACTION_CLAMP",
    "ACTION_REMOVE_ANNOTATION",
    "ACTION_REMOVE_IMAGE",
    "ACTION_REPORT",
    "RULES",
    "RULES_BY_ID",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_ORDER",
    "SEVERITY_WARNING",
    "CleanConfig",
    "CleanContext",
    "CleanReport",
    "Finding",
    "Rule",
    "clean",
    "list_rules",
]
