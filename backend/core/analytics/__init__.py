"""统计分析与质量报告模块。"""

from __future__ import annotations

from .analytics import AnalyticsConfig, AnalyticsReport, analyze, sample_images
from .charts import hbar_chart, heat_cell, vbar_chart
from .report import render_html_report, write_html_report

__all__ = [
    "AnalyticsConfig",
    "AnalyticsReport",
    "analyze",
    "hbar_chart",
    "heat_cell",
    "render_html_report",
    "sample_images",
    "vbar_chart",
    "write_html_report",
]
