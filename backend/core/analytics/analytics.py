"""数据集统计分析。

产出面向决策的统计，而不是一堆数字：

* **类别分布** —— 量化不均衡程度（最大/最小类实例数之比），指出哪些类别样本不足；
* **目标尺寸分布** —— 按 COCO 的小/中/大划分，直接对应 YOLO 的 P3/P4/P5 检测层，
  决定 `imgsz` 该不该调；
* **每图目标数** —— 反映场景密集程度，影响 mosaic 等增强的收益；
* **划分覆盖矩阵** —— 每类在每个子集是否都有样本（缺失会导致该指标 NaN）；
* **结论与建议** —— 由上列事实推导，明确可执行。

设计原则：**只报告能从数据算出的事实，不编造综合评分**。
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..ir import KIND_BBOX, KIND_IMAGE, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, DatasetBundle

_SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# COCO 的目标尺寸定义：小 <32²、中 32²~96²、大 >96²（单位：像素面积）
_SMALL_AREA = 32 * 32
_LARGE_AREA = 96 * 96

# 框面积占整图比例的分桶边界
_AREA_RATIO_EDGES = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0001)

# 每图目标数的分桶边界。首桶以 0 起，用于容纳「无标注图」——
# 若首边界为 1，值为 0 的图像会落不到任何区间，
# 被兜底逻辑错误地计入最后一桶（表现为"目标最多的那一档"）。
_OBJECTS_EDGES = (0, 1, 2, 3, 5, 10, 20, 50, 100000)

# 类别实例数低于该值时提示样本不足
_MIN_CLASS_INSTANCES = 20


@dataclass
class AnalyticsConfig:
    """统计分析参数。"""

    sample_size: int = 24              # 预览网格抽样张数
    max_image_sizes: int = 12          # 图像尺寸列表最多展示几种
    min_class_instances: int = _MIN_CLASS_INSTANCES


@dataclass
class AnalyticsReport:
    summary: Dict[str, Any] = field(default_factory=dict)
    class_distribution: List[Dict[str, Any]] = field(default_factory=list)
    class_by_split: Dict[str, Dict[str, int]] = field(default_factory=dict)
    size_category: List[Dict[str, Any]] = field(default_factory=list)
    area_ratio_histogram: Dict[str, Any] = field(default_factory=dict)
    objects_per_image_histogram: Dict[str, Any] = field(default_factory=dict)
    image_sizes: List[Dict[str, Any]] = field(default_factory=list)
    split_summary: List[Dict[str, Any]] = field(default_factory=list)
    imbalance: Dict[str, Any] = field(default_factory=dict)
    findings: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "class_distribution": self.class_distribution,
            "class_by_split": self.class_by_split,
            "size_category": self.size_category,
            "area_ratio_histogram": self.area_ratio_histogram,
            "objects_per_image_histogram": self.objects_per_image_histogram,
            "image_sizes": self.image_sizes,
            "split_summary": self.split_summary,
            "imbalance": self.imbalance,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "samples": self.samples,
        }


# ---------------------------------------------------------------------------


def analyze(
    bundle: DatasetBundle,
    config: Optional[AnalyticsConfig] = None,
) -> AnalyticsReport:
    """对数据集做统计分析。"""
    config = config or AnalyticsConfig()
    report = AnalyticsReport()

    images = list(bundle.images.values())
    annotations = bundle.annotations

    # ---------- 概览 ----------
    split_counts = {s: 0 for s in _SPLIT_ORDER}
    unassigned = 0
    for im in images:
        if im.split in split_counts:
            split_counts[im.split] += 1
        else:
            unassigned += 1

    num_bbox = sum(1 for a in annotations if a.kind == KIND_BBOX)
    num_image = len(annotations) - num_bbox

    report.summary = {
        "root": bundle.root,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_bbox_annotations": num_bbox,
        "num_image_labels": num_image,
        "annotation_kind": bundle.stats()["annotation_kind"],
        "num_classes": len(bundle.category_names()),
        "num_groups": len({im.group for im in images if im.group}),
        "split_counts": split_counts,
        "unassigned_images": unassigned,
    }

    # ---------- 类别分布 ----------
    counts = bundle.count_by_category()
    ordered = bundle.category_names()
    total_instances = sum(counts.get(n, 0) for n in ordered) or 1

    report.class_distribution = [
        {
            "name": name,
            "count": counts.get(name, 0),
            "share": round(counts.get(name, 0) / total_instances, 6),
        }
        for name in sorted(ordered, key=lambda n: -counts.get(n, 0))
    ]

    # 各类别在各子集的实例数
    report.class_by_split = {s: {n: 0 for n in ordered} for s in _SPLIT_ORDER}
    for im in images:
        split = im.split if im.split in _SPLIT_ORDER else None
        if split is None:
            continue
        for ann in bundle.annotations_of(im.uid):
            report.class_by_split[split][ann.category] = (
                report.class_by_split[split].get(ann.category, 0) + 1
            )

    report.imbalance = _imbalance(ordered, counts, config)

    # ---------- 目标尺寸（按 COCO 小/中/大） ----------
    size_counter: Counter = Counter()
    area_ratios: List[float] = []
    for im in images:
        w, h = float(im.width), float(im.height)
        image_area = w * h
        for ann in bundle.annotations_of(im.uid):
            if not ann.is_bbox or ann.bbox is None:
                continue
            area = ann.bbox.area
            size_counter[_size_category(area)] += 1
            if image_area > 0:
                area_ratios.append(area / image_area)

    report.size_category = [
        {"name": label, "count": size_counter.get(key, 0)}
        for key, label in (
            ("small", "小目标 (<32²)"),
            ("medium", "中目标 (32²~96²)"),
            ("large", "大目标 (>96²)"),
        )
    ]

    report.area_ratio_histogram = _histogram(
        area_ratios, _AREA_RATIO_EDGES, fmt="ratio"
    )

    # 每图目标数（含 0，无标注图也要计入分桶）
    per_image_counts = [len(bundle.annotations_of(im.uid)) for im in images]
    report.objects_per_image_histogram = _histogram(
        per_image_counts, _OBJECTS_EDGES, fmt="int"
    )

    # ---------- 图像尺寸 ----------
    size_pairs = Counter((im.width, im.height) for im in images)
    report.image_sizes = [
        {"size": f"{w}x{h}", "count": c}
        for (w, h), c in size_pairs.most_common(config.max_image_sizes)
    ]

    # ---------- 各子集概览 ----------
    for split in _SPLIT_ORDER:
        subset = [im for im in images if im.split == split]
        n_ann = sum(len(bundle.annotations_of(im.uid)) for im in subset)
        n_bbox = sum(
            1
            for im in subset
            for a in bundle.annotations_of(im.uid)
            if a.kind == KIND_BBOX
        )
        report.split_summary.append(
            {
                "split": split,
                "images": len(subset),
                "annotations": n_ann,
                "bbox_annotations": n_bbox,
                "classes_present": sum(
                    1
                    for name in ordered
                    if report.class_by_split[split].get(name, 0) > 0
                ),
            }
        )

    # ---------- 抽样预览 ----------
    report.samples = sample_images(bundle, config.sample_size)

    # ---------- 结论与建议 ----------
    _derive_findings(bundle, report, config)
    return report


# ---------------------------------------------------------------------------


def sample_images(bundle: DatasetBundle, limit: int = 24) -> List[Dict[str, Any]]:
    """抽样若干图像及其标注，供前端叠加预览。

    为让预览更有代表性，按类别轮流取样，而不是只取前 N 张
    （数据常按类别聚簇排列，只取前 N 张会只看到一两个类）。
    """
    if limit <= 0:
        return []

    by_class: Dict[str, List[str]] = OrderedDict()
    for name in bundle.category_names():
        by_class[name] = []

    unlabeled: List[str] = []
    for uid in bundle.images:
        anns = bundle.annotations_of(uid)
        if not anns:
            unlabeled.append(uid)
            continue
        for ann in anns:
            by_class.setdefault(ann.category, []).append(uid)
            break

    picked: List[str] = []
    seen = set()
    # 轮转各类别，保证每个类都出现
    while len(picked) < limit:
        progressed = False
        for name, uids in by_class.items():
            for uid in uids:
                if uid in seen:
                    continue
                seen.add(uid)
                picked.append(uid)
                progressed = True
                break
            if len(picked) >= limit:
                break
        if not progressed:
            break

    for uid in unlabeled:
        if len(picked) >= limit:
            break
        if uid not in seen:
            seen.add(uid)
            picked.append(uid)

    return [_image_payload(bundle, bundle.images[uid]) for uid in picked if uid in bundle.images]


def _image_payload(bundle: DatasetBundle, image) -> Dict[str, Any]:
    anns = bundle.annotations_of(image.uid)
    return {
        "uid": image.uid,
        "path": image.path,
        "rel_path": image.rel_path,
        "width": image.width,
        "height": image.height,
        "split": image.split,
        "group": image.group,
        "num_objects": len(anns),
        "objects": [
            {
                "category": a.category,
                "kind": a.kind,
                "bbox": list(a.bbox.as_tuple()) if a.bbox else None,
            }
            for a in anns
        ],
    }


# ---------------------------------------------------------------------------


def _size_category(area: float) -> str:
    if area < _SMALL_AREA:
        return "small"
    if area <= _LARGE_AREA:
        return "medium"
    return "large"


def _histogram(values: List[float], edges: Tuple[float, ...], fmt: str = "float") -> Dict[str, Any]:
    """按给定边界分桶，区间语义为 [lo, hi)。

    低于首个边界的值归入首桶，等于或高于末边界值的归入末桶，
    保证任何输入都不会被静默丢弃。
    """
    counts = [0] * (len(edges) - 1)
    for v in values:
        if v < edges[0]:
            counts[0] += 1
            continue
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if fmt == "ratio":
            labels.append(f"{lo * 100:.1f}~{hi * 100:.1f}%")
        elif fmt == "int":
            hi_label = f"{int(hi)}" if hi < 100000 else "+"
            labels.append(f"{int(lo)}~{hi_label}" if hi < 100000 else f"{int(lo)}+")
        else:
            labels.append(f"{lo}~{hi}")

    return {"labels": labels, "counts": counts, "edges": list(edges)}


def _imbalance(ordered: List[str], counts: Dict[str, int], config: AnalyticsConfig) -> Dict[str, Any]:
    """量化类别不均衡程度。"""
    present = [(n, counts.get(n, 0)) for n in ordered]
    if not present:
        return {}

    values = [c for _, c in present]
    total = sum(values) or 1
    top = max(present, key=lambda x: x[1])
    bottom = min(present, key=lambda x: x[1])

    # 前 20% 的类别占据了多少比例的实例（越高越不均衡）
    ordered_desc = sorted(values, reverse=True)
    k = max(1, int(len(ordered_desc) * 0.2))
    concentration = sum(ordered_desc[:k]) / total

    return {
        "max_class": top[0],
        "max_count": top[1],
        "min_class": bottom[0],
        "min_count": bottom[1],
        "ratio": round(top[1] / bottom[1], 2) if bottom[1] else None,
        "top20pct_share": round(concentration, 4),
        "underrepresented": [
            {"name": n, "count": c}
            for n, c in sorted(present, key=lambda x: x[1])
            if c < config.min_class_instances
        ],
    }


def _derive_findings(
    bundle: DatasetBundle,
    report: AnalyticsReport,
    config: AnalyticsConfig,
) -> None:
    """由上列事实推导结论与建议。只陈述可从数据验证的内容。"""
    recs = report.recommendations
    summary = report.summary

    # 类别不均衡
    imb = report.imbalance
    under = imb.get("underrepresented") or []
    if under:
        report.findings.append(
            {
                "level": "warning",
                "message": (
                    f"{len(under)} 个类别实例数少于 {config.min_class_instances}: "
                    + ", ".join(f"{u['name']}({u['count']})" for u in under[:8])
                ),
            }
        )
        recs.append(
            f"这 {len(under)} 个类别的样本量偏低，训练时它们的 mAP 会明显低于整体。"
            "优先补充这些类别的数据；若无法补充，可考虑合并语义相近的类别（类别规范化 → 合并）。"
        )

    if imb.get("ratio") and imb["ratio"] >= 10:
        recs.append(
            f"最大类与最小类实例数相差 {imb['ratio']} 倍（{imb['max_class']} vs {imb['min_class']}）。"
            "训练时建议开启类别加权，或对小类做过采样。"
        )

    # 目标尺寸
    small = next((s["count"] for s in report.size_category if s["name"].startswith("小")), 0)
    total_boxes = summary["num_bbox_annotations"] or 1
    if small / total_boxes > 0.5:
        recs.append(
            f"超过一半（{small / total_boxes:.0%}）的标注框属于小目标（面积 <32²）。"
            "小目标在默认 imgsz=640 下容易漏检，建议提高 imgsz（如 960/1280）或改用 P2 检测头。"
        )

    large = next((s["count"] for s in report.size_category if s["name"].startswith("大")), 0)
    if large / total_boxes > 0.8:
        recs.append(
            f"绝大多数（{large / total_boxes:.0%}）标注框是大目标。"
            "可以考虑降低 imgsz 以加快训练，精度损失有限。"
        )

    # 无标注图
    no_ann = sum(1 for im in bundle.images.values() if not bundle.annotations_of(im.uid))
    if no_ann:
        if summary["annotation_kind"] == KIND_IMAGE:
            report.findings.append(
                {"level": "error", "message": f"{no_ann} 张图没有任何标签，分类任务中属于无效样本"}
            )
            recs.append(f"这 {no_ann} 张无标签图应在清洗阶段删除（清洗规则 image_without_annotation）。")
        else:
            report.findings.append(
                {"level": "info", "message": f"{no_ann} 张图没有标注，可作背景负样本"}
            )
            recs.append("无标注图在检测任务中是合法的背景负样本，保留有助于降低误检。")

    # 划分覆盖
    if not summary["unassigned_images"]:
        missing_any = []
        for split in (SPLIT_VAL, SPLIT_TEST):
            absent = [
                name
                for name, count in (report.class_by_split.get(split) or {}).items()
                if count == 0
            ]
            if absent:
                missing_any.append((split, absent))
        for split, absent in missing_any:
            report.findings.append(
                {
                    "level": "warning",
                    "message": f"{split} 子集缺少 {len(absent)} 个类别: {', '.join(absent[:8])}",
                }
            )
        if missing_any:
            recs.append(
                "有类别在 val/test 中完全缺失，这些类别的指标会是 NaN。"
                "可调大固定比例、补充数据，或接受该类别暂不可评估。"
            )
    else:
        report.findings.append(
            {"level": "info", "message": f"{summary['unassigned_images']} 张图尚未划分"}
        )
        recs.append("导出前请先执行划分，否则未划分的图会全部归入 train。")

    # 子集类别构成是否与整体一致
    # 若 val 的类别比例与整体差异大，验证指标就代表不了真实分布，
    # 而这种问题不会体现在"每类是否都有样本"上，必须单独量化。
    overall_share = {c["name"]: c["share"] for c in report.class_distribution}
    shifts = []
    for split in (SPLIT_VAL, SPLIT_TEST):
        row = report.class_by_split.get(split) or {}
        split_total = sum(row.values())
        if split_total <= 0:
            continue
        for name, count in row.items():
            share = count / split_total
            base = overall_share.get(name, 0.0)
            delta = share - base
            if abs(delta) >= 0.10:
                shifts.append((split, name, share, base, delta))

    if shifts:
        shifts.sort(key=lambda x: -abs(x[4]))
        detail = "；".join(
            f"{sp} 的 {name} {share:.0%}（整体 {base:.0%}，{'偏高' if d > 0 else '偏低'} {abs(d):.0%}）"
            for sp, name, share, base, d in shifts[:4]
        )
        report.findings.append(
            {
                "level": "warning",
                "message": f"子集类别构成与整体不一致：{detail}",
            }
        )
        recs.append(
            "子集（尤其 val）的类别比例与整体差异较大，验证指标无法代表真实分布。"
            "建议重新划分并开启分层（stratified），让各子集的类别构成贴近整体。"
        )

    # 图像尺寸一致性
    if len(report.image_sizes) > 1:
        top = report.image_sizes[0]
        share = top["count"] / (summary["num_images"] or 1)
        if share < 0.8:
            report.findings.append(
                {
                    "level": "info",
                    "message": f"图像尺寸不统一，最常见的是 {top['size']}（占 {share:.0%}）",
                }
            )
            recs.append(
                "尺寸不一致时 YOLO 会统一缩放，长宽比差异大会引入黑边。"
                "若某一尺寸占绝对多数，可考虑统一到该尺寸以减少变形。"
            )
    elif report.image_sizes:
        report.findings.append(
            {"level": "info", "message": f"所有图像尺寸一致: {report.image_sizes[0]['size']}"}
        )

    if not recs:
        recs.append("未发现明显问题，数据分布健康。")
