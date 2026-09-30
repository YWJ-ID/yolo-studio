"""类别体系规范化。

解决三类问题：

1. **同义类名合并** —— `Closed Eye` / `closed_eye` / `closedeye` 本该是同一类，
   分散训练会让每类样本量都不足。提供自动建议 + 人工确认的两段式流程。

2. **类别取舍与重排** —— 保留/丢弃指定类别、显式指定类别顺序
   （用旧权重继续训练时类别顺序必须与旧模型一致）。

3. **按标注形态过滤** —— 检测框与图像级标注混合的数据集无法用单一目录布局导出
   （见 PROGRESS R-09）。这里提供「只保留某种形态」的能力，
   并自动清理因此变成无标注的图像。

所有操作都产出可回溯的报告，且与清洗一样：**不改原始数据文件**，只改内存中的统一 IR。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..ir import KIND_BBOX, KIND_IMAGE, Category, DatasetBundle

# 文件系统非法字符。分类布局会把类名当作目录名，含这些字符直接落盘失败。
_ILLEGAL_CHARS = set('<>:"/\\|?*')

# 类名长度上限（Windows 单层目录名限制 255，留足余量给子目录）
_MAX_NAME_LEN = 100


def normalize_key(name: str) -> str:
    """把类名归一成用于「判断是否同义」的键。

    只保留字母数字并转小写，因此 `Closed Eye` / `closed_eye` / `close_d-eye`
    都会得到 `closedeye`。注意这只是**用于比对**，绝不直接当作类名使用。
    """
    return re.sub(r"[^0-9a-z]+", "", str(name).lower())


def suggest_merges(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    """找出疑似同义但写法不同的类名，给出合并建议（不修改数据）。

    建议的规范名取「实例数最多的写法」——它是数据里实际使用最多的那个，
    合并到它名下改动最小，也最不容易让用户困惑。
    """
    counts = bundle.count_by_category()
    groups: Dict[str, List[str]] = defaultdict(list)
    for name in bundle.category_names():
        groups[normalize_key(name)].append(name)

    suggestions: List[Dict[str, Any]] = []
    for key, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        canonical = max(names, key=lambda n: (counts.get(n, 0), -len(n), n))
        suggestions.append(
            {
                "key": key,
                "names": names,
                "suggested": canonical,
                "counts": {n: counts.get(n, 0) for n in names},
                "total": sum(counts.get(n, 0) for n in names),
            }
        )
    # 合并收益大的排前面
    suggestions.sort(key=lambda s: -s["total"])
    return suggestions


def validate_class_names(names: List[str]) -> List[Dict[str, str]]:
    """检查类名是否可用作训练与落盘的类别名。"""
    issues: List[Dict[str, str]] = []

    for name in names:
        raw = "" if name is None else str(name)

        if not raw.strip():
            issues.append({"name": raw, "level": "error", "message": "类名为空或只有空白字符"})
            continue

        if raw != raw.strip():
            issues.append(
                {"name": raw, "level": "warning", "message": "类名首尾有空白字符，建议清理"}
            )

        bad = sorted(set(raw) & _ILLEGAL_CHARS)
        if bad:
            issues.append(
                {
                    "name": raw,
                    "level": "error",
                    "message": f"含文件系统非法字符 {bad}，分类布局落盘会失败",
                }
            )

        if raw in (".", ".."):
            issues.append({"name": raw, "level": "error", "message": "类名不能是 . 或 .."})

        if len(raw) > _MAX_NAME_LEN:
            issues.append(
                {
                    "name": raw,
                    "level": "warning",
                    "message": f"类名过长（{len(raw)} 字符），建议不超过 {_MAX_NAME_LEN}",
                }
            )

    # 归一化后重复 = 实际上会被当成同一类，容易混淆
    by_key: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        by_key[normalize_key(name)].append(name)
    for key, group in sorted(by_key.items()):
        if len(group) > 1:
            issues.append(
                {
                    "name": ", ".join(group),
                    "level": "warning",
                    "message": f"这些类名归一化后相同（'{key}'），建议合并为同一类",
                }
            )

    return issues


def sanitize_class_name(name: str, replacement: str = "_") -> str:
    """把类名改造成可安全用作目录名的形式。"""
    cleaned = "".join(
        replacement if (c in _ILLEGAL_CHARS or ord(c) < 32) else c for c in str(name)
    )
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned[:_MAX_NAME_LEN] or "class"


# ---------------------------------------------------------------------------


@dataclass
class TaxonomyConfig:
    """类别规范化配置。"""

    # 原名 -> 规范名。键不存在则保持原名
    mapping: Dict[str, str] = field(default_factory=dict)
    # 只保留这些类别（在其它的类别其标注会被删除）；为 None 表示不限制
    keep_classes: Optional[List[str]] = None
    # 丢弃这些类别（在 keep_classes 之后生效）
    drop_classes: List[str] = field(default_factory=list)
    # 只保留某种标注形态：None | "bbox" | "image"
    kind_filter: Optional[str] = None
    # 最终类别顺序（用于对齐旧模型的类别下标）
    class_order: Optional[List[str]] = None
    # 过滤后不再有任何标注的图像是否删除
    drop_empty_images: bool = True
    # 类名是否自动改造为安全目录名
    sanitize: bool = False


@dataclass
class TaxonomyReport:
    classes_before: List[str] = field(default_factory=list)
    classes_after: List[str] = field(default_factory=list)
    merged: Dict[str, str] = field(default_factory=dict)
    dropped_classes: List[str] = field(default_factory=list)
    """规范化后已无任何标注、因此被移出类别表的类别。"""
    classes_empty: List[str] = field(default_factory=list)
    removed_annotations: int = 0
    removed_by_kind: int = 0
    removed_by_class: int = 0
    removed_images: int = 0
    name_issues: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes_before": self.classes_before,
            "classes_after": self.classes_after,
            "merged": self.merged,
            "dropped_classes": self.dropped_classes,
            "classes_empty": self.classes_empty,
            "removed_annotations": self.removed_annotations,
            "removed_by_kind": self.removed_by_kind,
            "removed_by_class": self.removed_by_class,
            "removed_images": self.removed_images,
            "name_issues": self.name_issues,
            "warnings": self.warnings,
        }


def apply_taxonomy(
    bundle: DatasetBundle,
    config: Optional[TaxonomyConfig] = None,
) -> TaxonomyReport:
    """就地应用类别规范化，返回报告。"""
    config = config or TaxonomyConfig()
    report = TaxonomyReport()

    before = bundle.category_names()
    report.classes_before = list(before)

    mapping = dict(config.mapping or {})
    if config.sanitize:
        # 先 sanitize 映射结果，再补上未显式映射的类名
        for name in before:
            target = mapping.get(name, name)
            mapping[name] = sanitize_class_name(target)

    keep = set(config.keep_classes) if config.keep_classes is not None else None
    drop = set(config.drop_classes or ())

    if config.kind_filter not in (None, KIND_BBOX, KIND_IMAGE):
        raise ValueError(
            f"kind_filter 只能是 None / '{KIND_BBOX}' / '{KIND_IMAGE}'，"
            f"收到 {config.kind_filter!r}"
        )

    # ---------- 1. 逐条标注筛选 + 重命名 ----------
    kept_annotations = []
    dropped_names: Counter = Counter()

    for ann in bundle.annotations:
        # 形态过滤（解决混合数据集无法单布局导出的问题）
        if config.kind_filter and ann.kind != config.kind_filter:
            report.removed_by_kind += 1
            continue

        canonical = mapping.get(ann.category, ann.category)

        if keep is not None and canonical not in keep:
            report.removed_by_class += 1
            dropped_names[canonical] += 1
            continue
        if canonical in drop:
            report.removed_by_class += 1
            dropped_names[canonical] += 1
            continue

        if canonical != ann.category:
            report.merged[ann.category] = canonical
        ann.category = canonical
        kept_annotations.append(ann)

    report.removed_annotations = report.removed_by_kind + report.removed_by_class
    report.dropped_classes = sorted(dropped_names)

    bundle.annotations = kept_annotations
    bundle.invalidate_index()

    # ---------- 2. 重建类别表 ----------
    # 关键：**不能**只按原始顺序映射，必须剔除已无任何标注的类别。
    # 否则被丢弃的类会残留在类别表里，导出时生成 0 实例的类别下标
    # （分类布局下还会建出空目录），掩盖真实问题。
    kept = {a.category for a in bundle.annotations}
    explicit_order = {str(c) for c in (config.class_order or ())}

    new_order: List[str] = []
    empty_classes: List[str] = []
    for name in before:
        canonical = mapping.get(name, name)
        if canonical in new_order:
            continue
        if canonical not in kept and canonical not in explicit_order:
            empty_classes.append(canonical)
            continue
        new_order.append(canonical)

    report.classes_empty = sorted(empty_classes)
    if report.classes_empty:
        report.warnings.append(
            f"这些类别在规范化后已无任何标注，已从类别表中移除: {report.classes_empty}"
        )

    if config.class_order:
        requested = [str(c) for c in config.class_order]
        unknown = [
            c for c in requested if c not in new_order and c not in report.classes_before
        ]
        if unknown:
            report.warnings.append(
                f"class_order 中这些类别在数据里从未出现，已忽略: {unknown}"
            )
        ordered = [c for c in requested if c in new_order]
        ordered += [c for c in new_order if c not in ordered]
        new_order = ordered

    # 兜底：标注里出现了但没进 new_order 的类别（理论上不该发生）
    for ann in bundle.annotations:
        if ann.category not in new_order:
            new_order.append(ann.category)
            report.warnings.append(f"标注中出现了未登记的类别，已补入: {ann.category}")

    bundle.categories = {name: Category(name=name) for name in new_order}
    report.classes_after = list(new_order)

    # ---------- 3. 清理无标注图像 ----------
    if config.drop_empty_images and report.removed_annotations > 0:
        index = bundle.annotation_index()
        empties = [uid for uid in list(bundle.images) if not index.get(uid)]
        for uid in empties:
            bundle.remove_image(uid)
        report.removed_images = len(empties)

    # ---------- 4. 类名可落盘性检查 ----------
    report.name_issues = validate_class_names(report.classes_after)
    for issue in report.name_issues:
        if issue["level"] == "error":
            report.warnings.append(f"类名问题: {issue['name']} —— {issue['message']}")

    if not report.classes_after:
        report.warnings.append("规范化后没有剩下任何类别，该数据集将无法训练")

    return report
