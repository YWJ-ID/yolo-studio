"""生成自包含的 HTML 评估报告（M3-02）。

与数据质量报告（core/analytics/report.py）同一风格：单文件、样式内联、
图表用内联 SVG、ultralytics 生成的 PNG 以 base64 内嵌，可离线打开与分享。

只呈现能算出来的事实：逐类指标、混淆矩阵、耗时。
不做「模型评分」这类没有可验证含义的合成分数。
"""

from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..analytics.charts import hbar_chart, heat_cell
from .result import EvalResult

# 优先嵌入的图片：先看混淆矩阵与 PR 曲线，再看样例预测
_PREFERRED_IMAGES = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "val_batch0_pred.jpg",
    "val_batch0_labels.jpg",
)
_MAX_EMBED_BYTES = 3 * 1024 * 1024  # 单张超过 3MB 就不内嵌，避免报告过大

_CSS = """
*{box-sizing:border-box}
body{margin:0;padding:28px 32px;background:#f5f6f8;color:#262626;
     font-family:system-ui,-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
     font-size:13px;line-height:1.6}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 12px;padding-left:9px;border-left:3px solid #1677ff}
.meta{color:#8c8c8c;font-size:12px;margin-bottom:20px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.stat{flex:1 1 120px;background:#fff;border:1px solid #f0f0f0;border-radius:8px;padding:12px 14px}
.stat .k{color:#8c8c8c;font-size:12px}
.stat .v{font-size:20px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.card{background:#fff;border:1px solid #f0f0f0;border-radius:8px;padding:16px 18px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #f0f0f0;padding:6px 9px;text-align:left}
th{background:#fafafa;font-weight:600;color:#595959}
td.num{text-align:right;font-variant-numeric:tabular-nums}
svg{display:block;max-width:100%;height:auto}
.note{padding:10px 13px;border-radius:6px;background:#fffbe6;color:#d46b08;margin-bottom:8px}
.imgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.imgbox{background:#fff;border:1px solid #f0f0f0;border-radius:6px;overflow:hidden}
.imgbox img{display:block;width:100%;height:auto}
.imgbox .cap{padding:6px 8px;font-size:11px;color:#8c8c8c}
.legend{color:#8c8c8c;font-size:11px;margin-top:6px}
"""


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _embed_image(path: Path) -> Optional[str]:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_EMBED_BYTES:
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def _confusion_table(result: EvalResult) -> str:
    cm = result.confusion_matrix
    if not cm or not cm.get("matrix"):
        return ""
    labels: List[str] = list(cm.get("labels") or [])
    matrix: List[List[int]] = cm["matrix"]
    if not labels or not matrix:
        return ""

    max_value = max((v for row in matrix for v in row), default=0)
    head = "".join(f"<th>{escape(str(l))}</th>" for l in labels)
    rows = []
    for i, row in enumerate(matrix):
        name = labels[i] if i < len(labels) else f"class_{i}"
        cells = "".join(
            f'<td class="num" style="{heat_cell(int(v), max_value)}">{int(v)}</td>' for v in row
        )
        rows.append(f"<tr><th>{escape(str(name))}</th>{cells}</tr>")

    axis = escape(str(cm.get("axis") or "rows=预测,cols=真实"))
    return (
        '<div class="card">'
        f'<table><thead><tr><th>预测 \\ 真实</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        f'<div class="legend">方向：{axis}；对角线为正确分类，非对角为误判。</div>'
        "</div>"
    )


def render_eval_report(
    result: EvalResult,
    title: str = "模型评估报告",
    images_dir: Optional[Path] = None,
    embed_images: bool = True,
) -> str:
    """把评估结果渲染成单文件 HTML。"""
    sections: List[str] = []

    sections.append(
        f'<div class="meta">划分：{escape(result.split or "-")}　|　'
        f"模型：{escape(result.model_name or result.weights or '-')}　|　"
        f"任务：{escape(result.task or '-')}　|　"
        f"生成时间：{escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</div>"
    )

    # ---------- 概览 ----------
    cards = []
    for key, label in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "F1"),
        ("mAP50", "mAP50"),
        ("mAP50-95", "mAP50-95"),
        ("top1", "top1"),
        ("top5", "top5"),
    ):
        if key in result.overall:
            cards.append((label, _fmt(result.overall[key])))
    if result.per_class:
        cards.append(("类别数", str(len(result.per_class))))
    if cards:
        sections.append(
            '<div class="cards">'
            + "".join(
                f'<div class="stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
                for k, v in cards
            )
            + "</div>"
        )

    # ---------- 事实性说明 ----------
    notes: List[str] = []
    if result.data_yaml:
        notes.append(f"数据集：{result.data_yaml}")
    if result.weights:
        notes.append(f"权重：{result.weights}")
    if result.job_id:
        notes.append(f"来源训练任务：{result.job_id}")
    if result.per_class:
        empty = [c.name for c in result.per_class if c.instances == 0]
        if empty:
            notes.append(
                "以下类别在该划分中没有实例，其指标无统计意义："
                + "、".join(empty)
            )
    if result.speed:
        speed = "、".join(f"{k}={_fmt(v, 2)}ms" for k, v in result.speed.items())
        notes.append(f"单图耗时：{speed}")
    if notes:
        sections.append(
            "<h2>事实说明</h2>"
            + "".join(f'<div class="note">{escape(n)}</div>' for n in notes)
        )

    # ---------- 逐类指标 ----------
    if result.per_class:
        header = (
            "<tr><th>类别</th><th>实例数</th><th>precision</th><th>recall</th>"
            "<th>F1</th><th>AP50</th><th>AP50-95</th></tr>"
        )
        rows = "".join(
            f"<tr><td>{escape(c.name)}</td>"
            f'<td class="num">{c.instances}</td>'
            f'<td class="num">{_fmt(c.precision)}</td>'
            f'<td class="num">{_fmt(c.recall)}</td>'
            f'<td class="num">{_fmt(c.f1)}</td>'
            f'<td class="num">{_fmt(c.ap50)}</td>'
            f'<td class="num">{_fmt(c.ap50_95)}</td></tr>'
            for c in result.per_class
        )
        sections.append(
            "<h2>逐类指标</h2><div class='card'><table>"
            f"<thead>{header}</thead><tbody>{rows}</tbody></table></div>"
        )
        items = [(c.name, c.ap50_95) for c in sorted(result.per_class, key=lambda x: -x.ap50_95)]
        sections.append(
            "<h2>各类别 AP50-95</h2><div class='card'>"
            + hbar_chart(items, title="按 AP50-95 降序", value_fmt=lambda v: f"{v:.4f}")
            + "</div>"
        )

    # ---------- 混淆矩阵 ----------
    if result.confusion_matrix:
        sections.append("<h2>混淆矩阵</h2>" + _confusion_table(result))

    # ---------- 过程图像 ----------
    if embed_images and images_dir:
        blocks: List[Dict[str, Any]] = []
        for name in _PREFERRED_IMAGES:
            data_uri = _embed_image(Path(images_dir) / name)
            if data_uri:
                blocks.append({"name": name, "uri": data_uri})
        if blocks:
            html_blocks = "".join(
                f'<div class="imgbox"><img src="{b["uri"]}" alt="{escape(b["name"])}">'
                f'<div class="cap">{escape(b["name"])}</div></div>'
                for b in blocks
            )
            sections.append(f"<h2>过程图像</h2><div class='imgs'>{html_blocks}</div>")

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1>{body}</body></html>"
    )


def write_eval_report(
    result: EvalResult,
    out_path,
    title: str = "模型评估报告",
    images_dir=None,
    embed_images: bool = True,
) -> Path:
    """渲染并写出 HTML 报告，返回路径。"""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = render_eval_report(
        result,
        title=title,
        images_dir=Path(images_dir) if images_dir else None,
        embed_images=embed_images,
    )
    path.write_text(html, encoding="utf-8")
    return path
