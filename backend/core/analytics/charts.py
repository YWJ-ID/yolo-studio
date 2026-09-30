"""内联 SVG 图表。

为什么不用 matplotlib 或前端图表库：
    * 生成的 HTML 报告要能**离线打开、单文件分享**，不能依赖 CDN；
    * 图表数量少、样式简单，手写 SVG 比引入绘图依赖更轻。

前端页面则用 ECharts 渲染可交互图表，那份数据由 API 以 JSON 返回，
与本模块的静态图表互不干扰。
"""

from __future__ import annotations

from html import escape
from typing import Callable, Iterable, List, Sequence, Tuple

# 配色（与前端 ECharts 主题保持接近）
_PALETTE = [
    "#1677ff", "#52c41a", "#faad14", "#f5222d", "#13c2c2",
    "#722ed1", "#eb2f96", "#fa8c16", "#2f54eb", "#a0d911",
]


def hbar_chart(
    items: Sequence[Tuple[str, float]],
    title: str = "",
    width: int = 760,
    bar_height: int = 22,
    gap: int = 8,
    value_fmt: Callable[[float], str] = lambda v: f"{v:,.0f}",
) -> str:
    """水平条形图。类别名通常较长，横向排布更好读。"""
    if not items:
        return _empty(title)

    label_w = 190
    value_w = 80
    plot_w = max(60, width - label_w - value_w - 16)
    height = len(items) * (bar_height + gap) + 46

    max_value = max(v for _, v in items) or 1
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
    ]
    if title:
        parts.append(
            f'<text x="0" y="16" font-size="13" font-weight="600" fill="#1f1f1f">{escape(title)}</text>'
        )

    y = 34
    for i, (label, value) in enumerate(items):
        color = _PALETTE[i % len(_PALETTE)]
        bar_w = max(1.0, plot_w * (value / max_value)) if value > 0 else 0
        parts.append(
            f'<text x="0" y="{y + bar_height * 0.72}" font-size="12" fill="#434343">{escape(str(label))}</text>'
        )
        if bar_w > 0:
            parts.append(
                f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" '
                f'rx="3" fill="{color}" opacity="0.85"/>'
            )
        parts.append(
            f'<text x="{label_w + plot_w + 8}" y="{y + bar_height * 0.72}" '
            f'font-size="12" fill="#595959">{escape(value_fmt(value))}</text>'
        )
        y += bar_height + gap

    parts.append("</svg>")
    return "".join(parts)


def vbar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str = "",
    width: int = 760,
    height: int = 220,
    value_fmt: Callable[[float], str] = lambda v: f"{v:,.0f}",
) -> str:
    """垂直柱状图，适合分桶后的分布（标签短）。"""
    if not labels:
        return _empty(title)

    pad_l, pad_r, pad_t, pad_b = 44, 12, 34, 46
    plot_w = max(40, width - pad_l - pad_r)
    plot_h = max(40, height - pad_t - pad_b)

    max_value = max(values) or 1
    slot = plot_w / len(labels)
    bar_w = max(4.0, slot * 0.62)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
    ]
    if title:
        parts.append(
            f'<text x="0" y="16" font-size="13" font-weight="600" fill="#1f1f1f">{escape(title)}</text>'
        )

    # 横向网格线
    for i in range(5):
        gy = pad_t + plot_h * i / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="#f0f0f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" font-size="10" fill="#8c8c8c" '
            f'text-anchor="end">{value_fmt(max_value * (4 - i) / 4)}</text>'
        )

    for i, (label, value) in enumerate(zip(labels, values)):
        h = plot_h * (value / max_value) if value > 0 else 0
        x = pad_l + slot * i + (slot - bar_w) / 2
        y = pad_t + plot_h - h
        if h > 0:
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'rx="3" fill="{_PALETTE[i % len(_PALETTE)]}" opacity="0.85"/>'
            )
        # 标签按需旋转，避免重叠
        lx = pad_l + slot * i + slot / 2
        ly = pad_t + plot_h + 16
        if len(str(label)) > 6 and slot < 70:
            parts.append(
                f'<text x="{lx:.1f}" y="{ly}" font-size="10" fill="#595959" '
                f'text-anchor="end" transform="rotate(-35 {lx:.1f} {ly})">{escape(str(label))}</text>'
            )
        else:
            parts.append(
                f'<text x="{lx:.1f}" y="{ly}" font-size="10" fill="#595959" '
                f'text-anchor="middle">{escape(str(label))}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def heat_cell(value: int, max_value: int) -> str:
    """类别 × 子集矩阵的单元格底色。0 值用醒目的红底提示缺失。"""
    if max_value <= 0:
        return "background:#fafafa;color:#bfbfbf"
    if value == 0:
        return "background:#fff1f0;color:#cf1322;font-weight:600"
    ratio = value / max_value
    if ratio > 0.6:
        alpha = 0.30
    elif ratio > 0.3:
        alpha = 0.20
    elif ratio > 0.1:
        alpha = 0.12
    else:
        alpha = 0.06
    return f"background:rgba(22,119,255,{alpha:.2f})"


def _empty(title: str) -> str:
    label = escape(title) if title else ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="60">'
        f'<text x="0" y="30" font-size="12" fill="#8c8c8c">{label} 无数据</text></svg>'
    )
