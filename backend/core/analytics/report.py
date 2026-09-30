"""生成自包含的 HTML 质量报告。

产物是**单个 HTML 文件**：样式、图表（内联 SVG）、抽样缩略图（base64）全部内嵌，
不依赖任何外部资源，可以直接发给别人或归档留证。
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analytics import AnalyticsReport
from .charts import hbar_chart, heat_cell, vbar_chart

_LEVEL_STYLE = {
    "error": ("#fff1f0", "#cf1322", "严重"),
    "warning": ("#fffbe6", "#d46b08", "警告"),
    "info": ("#f6ffed", "#389e0d", "提示"),
}

_CSS = """
*{box-sizing:border-box}
body{margin:0;padding:28px 32px;background:#f5f6f8;color:#262626;
     font-family:system-ui,-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
     font-size:13px;line-height:1.6}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 12px;padding-left:9px;border-left:3px solid #1677ff}
.meta{color:#8c8c8c;font-size:12px;margin-bottom:20px}
.card{background:#fff;border:1px solid #f0f0f0;border-radius:8px;padding:16px 18px;margin-bottom:16px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.stat{flex:1 1 130px;background:#fff;border:1px solid #f0f0f0;border-radius:8px;padding:12px 14px}
.stat .k{color:#8c8c8c;font-size:12px}
.stat .v{font-size:20px;font-weight:600;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #f0f0f0;padding:6px 9px;text-align:left}
th{background:#fafafa;font-weight:600;color:#595959}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.alert{padding:10px 13px;border-radius:6px;margin-bottom:8px}
.recs{margin:0;padding-left:20px}
.recs li{margin-bottom:6px}
svg{display:block;max-width:100%;height:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:12px}
.thumb{background:#fff;border:1px solid #f0f0f0;border-radius:6px;overflow:hidden}
.thumb .img{position:relative;background:#fafafa}
.thumb img{display:block;width:100%;height:auto}
.thumb .cap{padding:5px 7px;font-size:11px;color:#8c8c8c;
            overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.thumb .tag{display:inline-block;padding:0 5px;border-radius:3px;background:#f0f5ff;color:#1677ff;font-size:10px}
.legend{color:#8c8c8c;font-size:11px;margin-top:6px}
"""


def render_html_report(
    report: AnalyticsReport,
    title: str = "数据集质量报告",
    embed_samples: bool = True,
    thumb_width: int = 200,
) -> str:
    """把分析结果渲染成单文件 HTML。"""
    s = report.summary
    sections: List[str] = []

    # ---------- 概览 ----------
    cards = [
        ("图像", s.get("num_images", 0)),
        ("标注", s.get("num_annotations", 0)),
        ("类别", s.get("num_classes", 0)),
        ("分组", s.get("num_groups", 0)),
    ]
    for split, label in (("train", "train"), ("val", "val"), ("test", "test")):
        cards.append((label, (s.get("split_counts") or {}).get(split, 0)))
    sections.append(
        '<div class="cards">'
        + "".join(
            f'<div class="stat"><div class="k">{escape(k)}</div><div class="v">{v:,}</div></div>'
            for k, v in cards
        )
        + "</div>"
    )

    kind_label = {
        "bbox": "目标检测（带边界框）",
        "image": "图像分类（图像级标签）",
        "mixed": "混合形态",
        "unknown": "无标注",
    }.get(s.get("annotation_kind", ""), s.get("annotation_kind", ""))
    sections.append(
        f'<div class="card"><b>标注形态：</b>{escape(str(kind_label))}　'
        f'<b>检测框：</b>{s.get("num_bbox_annotations", 0):,}　'
        f'<b>图像级标签：</b>{s.get("num_image_labels", 0):,}</div>'
    )

    # ---------- 结论 ----------
    if report.findings:
        blocks = []
        for f in report.findings:
            level = f.get("level", "info")
            bg, fg, label = _LEVEL_STYLE.get(level, _LEVEL_STYLE["info"])
            blocks.append(
                f'<div class="alert" style="background:{bg};color:{fg}">'
                f'<b>[{label}]</b> {escape(f.get("message", ""))}</div>'
            )
        sections.append("<h2>检查结论</h2>" + "".join(blocks))

    # ---------- 建议 ----------
    if report.recommendations:
        items = "".join(f"<li>{escape(r)}</li>" for r in report.recommendations)
        sections.append(f'<h2>处置建议</h2><div class="card"><ul class="recs">{items}</ul></div>')

    # ---------- 类别分布 ----------
    if report.class_distribution:
        items = [(c["name"], c["count"]) for c in report.class_distribution]
        sections.append(
            "<h2>类别分布</h2><div class='card'>"
            + hbar_chart(items, title=f"共 {len(items)} 个类别")
            + _imbalance_note(report)
            + "</div>"
        )

    # ---------- 类别 × 子集 ----------
    if report.class_by_split:
        max_cell = max(
            (v for row in report.class_by_split.values() for v in row.values()),
            default=0,
        )
        head = "".join(
            f"<th>{escape(sp)}</th>" for sp in ("train", "val", "test")
        )
        rows = []
        for cls, _ in [(c["name"], c["count"]) for c in report.class_distribution]:
            cells = []
            for sp in ("train", "val", "test"):
                v = (report.class_by_split.get(sp) or {}).get(cls, 0)
                cells.append(
                    f'<td class="num" style="{heat_cell(v, max_cell)}">{v:,}</td>'
                )
            rows.append(f"<tr><td>{escape(cls)}</td>{''.join(cells)}</tr>")
        sections.append(
            "<h2>类别 × 子集</h2><div class='card'><table>"
            f"<tr><th>类别</th>{head}</tr>{''.join(rows)}</table>"
            "<div class='legend'>红色 = 该类别在此子集中完全缺失，对应指标会是 NaN</div></div>"
        )

    # ---------- 目标尺寸 ----------
    if report.size_category:
        labels = [x["name"] for x in report.size_category]
        values = [x["count"] for x in report.size_category]
        sections.append(
            "<h2>目标尺寸分布</h2><div class='card'>"
            + vbar_chart(labels, values, title="按 COCO 尺寸划分（对应 P3/P4/P5 检测层）")
            + "</div>"
        )

    # ---------- 框面积占比 ----------
    h = report.area_ratio_histogram
    if h.get("counts"):
        sections.append(
            "<h2>标注框面积占比</h2><div class='card'>"
            + vbar_chart(h["labels"], h["counts"], title="框面积 / 整图面积")
            + "</div>"
        )

    # ---------- 每图目标数 ----------
    h2 = report.objects_per_image_histogram
    if h2.get("counts"):
        sections.append(
            "<h2>每图目标数</h2><div class='card'>"
            + vbar_chart(h2["labels"], h2["counts"], title="反映场景密集程度")
            + "</div>"
        )

    # ---------- 图像尺寸 / 子集概览 ----------
    if report.image_sizes:
        rows = "".join(
            f'<tr><td>{escape(x["size"])}</td><td class="num">{x["count"]:,}</td></tr>'
            for x in report.image_sizes
        )
        sections.append(
            "<h2>图像尺寸</h2><div class='card'><table>"
            f"<tr><th>尺寸</th><th>数量</th></tr>{rows}</table></div>"
        )

    if report.split_summary:
        rows = "".join(
            f'<tr><td>{escape(x["split"])}</td>'
            f'<td class="num">{x["images"]:,}</td>'
            f'<td class="num">{x["annotations"]:,}</td>'
            f'<td class="num">{x["bbox_annotations"]:,}</td>'
            f'<td class="num">{x["classes_present"]}</td></tr>'
            for x in report.split_summary
        )
        sections.append(
            "<h2>子集概览</h2><div class='card'><table>"
            "<tr><th>子集</th><th>图像</th><th>标注</th><th>检测框</th><th>覆盖类别数</th></tr>"
            f"{rows}</table></div>"
        )

    # ---------- 抽样预览 ----------
    if report.samples:
        thumbs = []
        for sample in report.samples:
            src = _thumbnail(sample.get("path", ""), thumb_width) if embed_samples else ""
            if not src:
                continue
            tags = "".join(
                f'<span class="tag">{escape(o["category"])}</span> '
                for o in _unique_categories(sample)
            )
            thumbs.append(
                '<div class="thumb">'
                f'<div class="img"><img src="{src}" alt=""></div>'
                f'<div class="cap" title="{escape(sample.get("rel_path", ""))}">'
                f"{escape(sample.get('rel_path', ''))}</div>"
                f'<div class="cap">{tags}</div>'
                "</div>"
            )
        if thumbs:
            sections.append(
                f"<h2>抽样预览（{len(thumbs)} 张）</h2>"
                f"<div class='card'><div class='grid'>{''.join(thumbs)}</div>"
                "<div class='legend'>样本按类别轮转抽取，保证每个类别都有代表</div></div>"
            )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"<div class='meta'>生成时间 {generated}　·　数据来源 "
        f"{escape(str(s.get('root', '') or '（多来源合并）'))}</div>"
        + "".join(sections)
        + "</body></html>"
    )


def write_html_report(
    report: AnalyticsReport,
    out_path,
    title: str = "数据集质量报告",
    embed_samples: bool = True,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html_report(report, title=title, embed_samples=embed_samples),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------


def _imbalance_note(report: AnalyticsReport) -> str:
    imb = report.imbalance or {}
    if not imb:
        return ""
    ratio = imb.get("ratio")
    share = imb.get("top20pct_share")
    bits = []
    if ratio:
        bits.append(
            f"最大类 {imb['max_class']}({imb['max_count']:,}) 与最小类 "
            f"{imb['min_class']}({imb['min_count']:,}) 相差 <b>{ratio}×</b>"
        )
    if share is not None:
        bits.append(f"实例数前 20% 的类别占据了 <b>{share:.0%}</b> 的标注")
    under = imb.get("underrepresented") or []
    if under:
        bits.append(f"{len(under)} 个类别实例数偏少")
    return f"<div class='legend'>{'　·　'.join(bits)}</div>" if bits else ""


def _unique_categories(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen = {}
    for obj in sample.get("objects", []):
        seen.setdefault(obj.get("category"), obj)
    return list(seen.values())


def _thumbnail(path: str, width: int, quality: int = 65) -> str:
    """把图像缩成 base64 JPEG，让报告不依赖外部文件。"""
    if not path:
        return ""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > width:
                ratio = width / im.width
                im = im.resize((width, max(1, int(im.height * ratio))), Image.LANCZOS)
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""
