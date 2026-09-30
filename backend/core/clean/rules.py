"""清洗规则集合。

分为四类：
    * 图像类    —— 损坏、尺寸异常、方向标记
    * 重复类    —— 精确重复（sha1）、近似重复（pHash）
    * 标签类    —— 越界、零面积、重复框、类别名变体
    * 泄漏类    —— 跨子集重复、同组跨子集

每条规则返回 Finding 列表。规则本身**不改数据**，处置由 engine 统一执行，
这样 dry-run 与正式清洗走的是完全相同的判定逻辑。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..io import find_near_duplicates, get_exif_orientation, is_readable, phash_bits, sha1_file
from ..ir import KIND_BBOX, KIND_IMAGE, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, DatasetBundle, ImageRecord
from .findings import (
    ACTION_CLAMP,
    ACTION_REMOVE_ANNOTATION,
    ACTION_REMOVE_IMAGE,
    ACTION_REPORT,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Finding,
)

REAL_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# 浮点比较容差：归一化坐标往返会产生 ~1e-6 级别误差，不能当越界
_EPS = 1e-3

# 保留优先级：优先保留 train，其次 val/test，未划分的最后
_SPLIT_RANK = {SPLIT_TRAIN: 0, SPLIT_VAL: 1, SPLIT_TEST: 2}


@dataclass
class CleanConfig:
    """清洗参数。"""

    # --- 图像 ---
    verify_readable: bool = True          # 完整解码校验（较慢但最可靠）
    min_width: int = 32
    min_height: int = 32
    max_aspect_ratio: float = 20.0
    check_exif: bool = True

    # --- 重复 ---
    check_exact_duplicates: bool = True
    check_near_duplicates: bool = True
    near_duplicate_distance: int = 6      # pHash 汉明距离阈值，越小越严格

    # --- 标签 ---
    max_box_area_ratio: float = 0.95      # 超过整图 95% 的框提示复核
    min_box_area_ratio: float = 0.0       # 0 = 关闭
    check_duplicate_boxes: bool = True

    # --- 类别 ---
    min_class_instances: int = 5

    # --- 其他 ---
    limit: int = 0                        # >0 时只检查前 N 张（用于快速预览）
    max_findings_per_rule: int = 1000     # 单条规则最多保留多少条明细
    disabled_rules: List[str] = field(default_factory=list)


class CleanContext:
    """规则运行上下文：承载惰性计算结果，避免规则之间重复计算。"""

    def __init__(self, bundle: DatasetBundle, config: CleanConfig, report):
        self.bundle = bundle
        self.config = config
        self.report = report
        self.all_images: List[ImageRecord] = list(bundle.images.values())
        self.images: List[ImageRecord] = (
            self.all_images[: config.limit] if config.limit > 0 else self.all_images
        )
        self._sha1_groups: Optional[Dict[str, List[str]]] = None
        self._phash_entries: Optional[List[Tuple[str, str]]] = None

    # ---------- 惰性哈希 ----------

    def sha1_groups(self) -> Dict[str, List[str]]:
        if self._sha1_groups is not None:
            return self._sha1_groups

        groups: Dict[str, List[str]] = defaultdict(list)
        computed = 0
        for im in self.images:
            digest = im.sha1 or sha1_file(im.path)
            if not digest:
                continue
            if not im.sha1:
                im.sha1 = digest
                computed += 1
            groups[digest].append(im.uid)
        # 已带哈希的图也计入统计
        self.report.hashed_sha1 = len(groups)
        if computed:
            self.report.warnings.append(f"本次计算了 {computed} 张图的 sha1")
        self._sha1_groups = dict(groups)
        return self._sha1_groups

    def phash_entries(self) -> List[Tuple[str, str]]:
        if self._phash_entries is not None:
            return self._phash_entries

        entries: List[Tuple[str, str]] = []
        computed = 0
        for im in self.images:
            digest = im.phash or phash_bits(im.path)
            if not digest:
                continue
            if not im.phash:
                im.phash = digest
                computed += 1
            entries.append((im.uid, digest))
        self.report.hashed_phash = len(entries)
        if computed:
            self.report.warnings.append(f"本次计算了 {computed} 张图的 pHash")
        self._phash_entries = entries
        return self._phash_entries


# ---------------------------------------------------------------------------
# 图像类
# ---------------------------------------------------------------------------


def rule_image_unreadable(ctx: CleanContext) -> List[Finding]:
    if not ctx.config.verify_readable:
        return []
    out = []
    for im in ctx.images:
        if not is_readable(im.path):
            out.append(
                Finding(
                    rule="image_unreadable",
                    severity=SEVERITY_ERROR,
                    message="图像无法解码（文件损坏或非图像）",
                    action=ACTION_REMOVE_IMAGE,
                    image=im,
                    image_uid=im.uid,
                    image_path=im.path,
                )
            )
    return out


def rule_image_invalid_size(ctx: CleanContext) -> List[Finding]:
    out = []
    for im in ctx.images:
        if im.width <= 0 or im.height <= 0:
            out.append(
                Finding(
                    rule="image_invalid_size",
                    severity=SEVERITY_ERROR,
                    message="无法读取图像尺寸，标注坐标无从校验",
                    action=ACTION_REMOVE_IMAGE,
                    image=im,
                    image_uid=im.uid,
                    image_path=im.path,
                    detail={"width": im.width, "height": im.height},
                )
            )
    return out


def rule_image_too_small(ctx: CleanContext) -> List[Finding]:
    out = []
    for im in ctx.images:
        if im.width <= 0 or im.height <= 0:
            continue
        if im.width < ctx.config.min_width or im.height < ctx.config.min_height:
            out.append(
                Finding(
                    rule="image_too_small",
                    severity=SEVERITY_WARNING,
                    message=f"图像过小 {im.width}x{im.height}（阈值 "
                    f"{ctx.config.min_width}x{ctx.config.min_height}）",
                    action=ACTION_REPORT,
                    image=im,
                    image_uid=im.uid,
                    image_path=im.path,
                    detail={"width": im.width, "height": im.height},
                )
            )
    return out


def rule_image_extreme_aspect(ctx: CleanContext) -> List[Finding]:
    out = []
    limit = ctx.config.max_aspect_ratio
    if limit <= 0:
        return out
    for im in ctx.images:
        if im.width <= 0 or im.height <= 0:
            continue
        ratio = max(im.width, im.height) / max(1, min(im.width, im.height))
        if ratio > limit:
            out.append(
                Finding(
                    rule="image_extreme_aspect",
                    severity=SEVERITY_INFO,
                    message=f"宽高比异常 {ratio:.1f}:1（阈值 {limit}:1）",
                    action=ACTION_REPORT,
                    image=im,
                    image_uid=im.uid,
                    image_path=im.path,
                    detail={"ratio": round(ratio, 2), "width": im.width, "height": im.height},
                )
            )
    return out


def rule_image_exif_orientation(ctx: CleanContext) -> List[Finding]:
    """EXIF 方向标记不为 1 时，图片实际显示方向与像素方向不一致，
    标注框会整体错位。手机/相机拍摄的数据常见。"""
    if not ctx.config.check_exif:
        return []
    out = []
    for im in ctx.images:
        orientation = get_exif_orientation(im.path)
        if orientation and orientation != 1:
            out.append(
                Finding(
                    rule="image_exif_orientation",
                    severity=SEVERITY_WARNING,
                    message=f"EXIF 方向标记为 {orientation}，标注框可能整体错位",
                    action=ACTION_REPORT,
                    image=im,
                    image_uid=im.uid,
                    image_path=im.path,
                    detail={"orientation": orientation},
                )
            )
    return out


# ---------------------------------------------------------------------------
# 重复类
# ---------------------------------------------------------------------------


def rule_duplicate_exact(ctx: CleanContext) -> List[Finding]:
    """完全相同的图（按内容哈希）。

    跨子集重复 = 数据泄漏，会让验证指标虚高，必须删；
    同子集内重复 = 冗余，删掉多余的（保留优先级 train > val > test）。
    """
    if not ctx.config.check_exact_duplicates:
        return []

    out: List[Finding] = []
    for digest, uids in ctx.sha1_groups().items():
        if len(uids) < 2:
            continue

        images = [ctx.bundle.images[u] for u in uids if u in ctx.bundle.images]
        if len(images) < 2:
            continue
        images.sort(key=lambda im: (_SPLIT_RANK.get(im.split, 9), im.uid))
        keep = images[0]
        present_splits = {im.split for im in images if im.split in REAL_SPLITS}
        leaked = len(present_splits) > 1

        for im in images[1:]:
            out.append(
                Finding(
                    rule="leak_duplicate_across_splits" if leaked else "image_duplicate_exact",
                    severity=SEVERITY_ERROR if leaked else SEVERITY_WARNING,
                    message=(
                        f"与其它子集存在完全相同的图（sha1 {digest[:10]}），"
                        f"会造成验证指标虚高"
                        if leaked
                        else f"完全重复的图（sha1 {digest[:10]}）"
                    ),
                    action=ACTION_REMOVE_IMAGE,
                    image=im,
                    image_uid=im.uid,
                    image_path=im.path,
                    detail={
                        "sha1": digest,
                        "duplicate_of": keep.path,
                        "split": im.split,
                        "kept_split": keep.split,
                        "splits": sorted(str(s) for s in present_splits),
                        "group_size": len(images),
                    },
                )
            )
    return out


def rule_duplicate_near(ctx: CleanContext) -> List[Finding]:
    """近似重复（裁剪/压缩/轻微改动）。

    这类**不自动删除**——相邻视频帧本来就高度相似，但它们是合法且不同的样本。
    仅报告，由人工判断；跨子集的则必须警惕。
    """
    if not ctx.config.check_near_duplicates:
        return []

    entries = ctx.phash_entries()
    pairs = find_near_duplicates(entries, ctx.config.near_duplicate_distance)
    ctx.report.near_duplicate_pairs = len(pairs)

    out: List[Finding] = []
    for uid_a, uid_b in pairs:
        a = ctx.bundle.images.get(uid_a)
        b = ctx.bundle.images.get(uid_b)
        if not a or not b:
            continue
        leaked = (
            a.split in REAL_SPLITS
            and b.split in REAL_SPLITS
            and a.split != b.split
        )
        out.append(
            Finding(
                rule="leak_near_duplicate_across_splits" if leaked else "image_duplicate_near",
                severity=SEVERITY_ERROR if leaked else SEVERITY_INFO,
                message=(
                    f"与另一子集中的图高度相似（{b.split} vs {a.split}），疑似泄漏"
                    if leaked
                    else "与另一张图高度相似（可能重复或为相邻帧）"
                ),
                action=ACTION_REPORT,
                image=b,
                image_uid=b.uid,
                image_path=b.path,
                detail={
                    "similar_to": a.path,
                    "split": b.split,
                    "similar_to_split": a.split,
                    "algorithm": f"phash-hamming<={ctx.config.near_duplicate_distance}",
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# 泄漏类
# ---------------------------------------------------------------------------


def rule_leak_group_across_splits(ctx: CleanContext) -> List[Finding]:
    """同一分组（同视频/序列）被拆到了不同子集。

    输入数据集常常自带划分，而那份划分未必按视频切分——
    视频抽帧数据一旦这样泄漏，验证指标会严重虚高。
    """
    by_group: Dict[str, set] = defaultdict(set)
    for im in ctx.images:
        if im.group and im.split in REAL_SPLITS:
            by_group[im.group].add(im.split)

    out = []
    for group, splits in sorted(by_group.items()):
        if len(splits) > 1:
            out.append(
                Finding(
                    rule="leak_group_across_splits",
                    severity=SEVERITY_ERROR,
                    message=(
                        f"分组 '{group}' 横跨多个子集 {sorted(splits)}，属于数据泄漏。"
                        "解决方式是重新划分（开启分组防泄漏），而不是删除数据，"
                        "因此本规则只报告、不自动处置。"
                    ),
                    action=ACTION_REPORT,
                    detail={
                        "group": group,
                        "splits": sorted(splits),
                        "fix": "重新执行划分并保持 respect_groups=True",
                    },
                )
            )
    return out


# ---------------------------------------------------------------------------
# 标签类
# ---------------------------------------------------------------------------


def rule_label_out_of_bounds(ctx: CleanContext) -> List[Finding]:
    out = []
    for im in ctx.images:
        w, h = float(im.width), float(im.height)
        if w <= 0 or h <= 0:
            continue
        for ann in ctx.bundle.annotations_of(im.uid):
            b = ann.bbox
            if b is None:
                continue
            if (
                b.x1 < -_EPS
                or b.y1 < -_EPS
                or b.x2 > w + _EPS
                or b.y2 > h + _EPS
            ):
                out.append(
                    Finding(
                        rule="label_out_of_bounds",
                        severity=SEVERITY_WARNING,
                        message=(
                            f"标注框超出图像范围 "
                            f"({b.x1:.1f},{b.y1:.1f},{b.x2:.1f},{b.y2:.1f}) 图 {w:.0f}x{h:.0f}"
                        ),
                        action=ACTION_CLAMP,
                        image=im,
                        annotation=ann,
                        image_uid=im.uid,
                        image_path=im.path,
                        category=ann.category,
                        detail={
                            "bbox": list(b.as_tuple()),
                            "image_size": [im.width, im.height],
                        },
                    )
                )
    return out


def rule_label_invalid_size(ctx: CleanContext) -> List[Finding]:
    out = []
    for im in ctx.images:
        for ann in ctx.bundle.annotations_of(im.uid):
            b = ann.bbox
            if b is None:
                continue
            if b.width <= _EPS or b.height <= _EPS:
                out.append(
                    Finding(
                        rule="label_invalid_size",
                        severity=SEVERITY_ERROR,
                        message=f"标注框宽或高为 0（宽 {b.width:.3f} 高 {b.height:.3f}）",
                        action=ACTION_REMOVE_ANNOTATION,
                        image=im,
                        annotation=ann,
                        image_uid=im.uid,
                        image_path=im.path,
                        category=ann.category,
                        detail={"bbox": list(b.as_tuple())},
                    )
                )
    return out


def rule_label_box_area(ctx: CleanContext) -> List[Finding]:
    """整图框（漏标背景）或极小框（误标）。"""
    out = []
    max_ratio = ctx.config.max_box_area_ratio
    min_ratio = ctx.config.min_box_area_ratio
    for im in ctx.images:
        w, h = float(im.width), float(im.height)
        if w <= 0 or h <= 0:
            continue
        image_area = w * h
        for ann in ctx.bundle.annotations_of(im.uid):
            b = ann.bbox
            if b is None:
                continue
            ratio = b.area / image_area
            if max_ratio > 0 and ratio > max_ratio:
                out.append(
                    Finding(
                        rule="label_box_too_large",
                        severity=SEVERITY_WARNING,
                        message=f"标注框占整图 {ratio:.1%}，疑似漏标背景/整图框",
                        action=ACTION_REPORT,
                        image=im,
                        annotation=ann,
                        image_uid=im.uid,
                        image_path=im.path,
                        category=ann.category,
                        detail={"area_ratio": round(ratio, 4)},
                    )
                )
            elif min_ratio > 0 and 0 < ratio < min_ratio:
                out.append(
                    Finding(
                        rule="label_box_too_small",
                        severity=SEVERITY_INFO,
                        message=f"标注框仅占整图 {ratio:.3%}，疑似误标",
                        action=ACTION_REPORT,
                        image=im,
                        annotation=ann,
                        image_uid=im.uid,
                        image_path=im.path,
                        category=ann.category,
                        detail={"area_ratio": round(ratio, 6)},
                    )
                )
    return out


def rule_label_duplicate(ctx: CleanContext) -> List[Finding]:
    """同一张图里类别与坐标完全相同的重复标注。"""
    if not ctx.config.check_duplicate_boxes:
        return []

    out = []
    for im in ctx.images:
        seen: Dict[Tuple[str, Tuple[float, float, float, float]], int] = {}
        for ann in ctx.bundle.annotations_of(im.uid):
            if ann.bbox is None:
                continue
            key = (
                ann.category,
                tuple(round(v, 3) for v in ann.bbox.as_tuple()),
            )
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                out.append(
                    Finding(
                        rule="label_duplicate",
                        severity=SEVERITY_WARNING,
                        message="同一张图存在类别与坐标完全相同的重复标注",
                        action=ACTION_REMOVE_ANNOTATION,
                        image=im,
                        annotation=ann,
                        image_uid=im.uid,
                        image_path=im.path,
                        category=ann.category,
                        detail={"bbox": list(ann.bbox.as_tuple())},
                    )
                )
    return out


def rule_image_without_annotation(ctx: CleanContext) -> List[Finding]:
    """无任何标注的图。

    检测任务中这是合法的「背景负样本」；分类任务中则是无效样本。
    """
    out = []
    for im in ctx.images:
        if ctx.bundle.annotations_of(im.uid):
            continue
        out.append(
            Finding(
                rule="image_without_annotation",
                severity=SEVERITY_INFO,
                message="图像没有任何标注（检测任务中可作背景负样本，分类任务中是无效样本）",
                action=ACTION_REPORT,
                image=im,
                image_uid=im.uid,
                image_path=im.path,
                detail={"split": im.split},
            )
        )
    return out


def rule_category_name_variants(ctx: CleanContext) -> List[Finding]:
    """疑似同义但写法不同的类别名。

    如 `Closed Eye` / `closed_eye` / `closedeye` —— 它们本该是同一类，
    分散训练会让每类样本量都偏少。交给 taxonomy 模块（M1-07）合并。
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for name in ctx.bundle.category_names():
        key = re.sub(r"[^0-9a-z]+", "", str(name).lower())
        groups[key].append(name)

    out = []
    for key, names in sorted(groups.items()):
        if len(names) > 1:
            out.append(
                Finding(
                    rule="category_name_variants",
                    severity=SEVERITY_WARNING,
                    message=f"疑似同义类别名（归一化后同为 '{key}'）: {names}",
                    action=ACTION_REPORT,
                    category=names[0],
                    detail={"normalized": key, "names": names},
                )
            )
    return out


def rule_class_too_few(ctx: CleanContext) -> List[Finding]:
    """实例数过少的类别，训练时几乎学不到，且会让指标剧烈波动。"""
    threshold = ctx.config.min_class_instances
    if threshold <= 0:
        return []
    out = []
    for name, count in sorted(ctx.bundle.count_by_category().items()):
        if count < threshold:
            out.append(
                Finding(
                    rule="class_too_few",
                    severity=SEVERITY_WARNING,
                    message=f"类别 '{name}' 实例数仅 {count}（阈值 {threshold}）",
                    action=ACTION_REPORT,
                    category=name,
                    detail={"count": count, "threshold": threshold},
                )
            )
    return out


def rule_orphan_labels(ctx: CleanContext) -> List[Finding]:
    """有标签文件却没有对应图像（由接入层反向扫描发现）。"""
    orphans = ctx.bundle.meta.get("orphan_labels") or []
    return [
        Finding(
            rule="label_orphan",
            severity=SEVERITY_WARNING,
            message="存在标签文件但找不到对应图像",
            action=ACTION_REPORT,
            detail={"label_file": str(path)},
        )
        for path in orphans
    ]


# ---------------------------------------------------------------------------
# 规则注册表
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    id: str
    title: str
    category: str
    description: str
    fn: Any

    def check(self, ctx: CleanContext) -> List[Finding]:
        return self.fn(ctx)


RULES: List[Rule] = [
    Rule("image_unreadable", "图像损坏", "图像",
         "无法解码的文件（下载中断、存储损坏）", rule_image_unreadable),
    Rule("image_invalid_size", "尺寸不可读", "图像",
         "读不出宽高，标注坐标无从校验", rule_image_invalid_size),
    Rule("image_too_small", "图像过小", "图像",
         "小于阈值的图，缩放后目标几乎不可见", rule_image_too_small),
    Rule("image_extreme_aspect", "宽高比异常", "图像",
         "极端长宽比，通常来自截图或拼接错误", rule_image_extreme_aspect),
    Rule("image_exif_orientation", "EXIF 方向", "图像",
         "方向标记不为 1，标注框会整体错位", rule_image_exif_orientation),

    # 注意：重复检测同时负责发现「跨子集重复」这一泄漏场景。
    # 泄漏变体的明细会以 leak_* 作为 rule id 出现，但不单独注册——
    # 否则同一个检查函数会被执行两次，产出重复明细。
    # 泄漏检测也不应被关闭，因此不提供独立开关。
    Rule("image_duplicate_exact", "完全重复图 / 跨子集泄漏", "重复",
         "内容完全相同（sha1）；跨子集重复即数据泄漏", rule_duplicate_exact),
    Rule("image_duplicate_near", "近似重复图 / 跨子集泄漏", "重复",
         "裁剪或压缩后高度相似；跨子集即为泄漏", rule_duplicate_near),

    Rule("leak_group_across_splits", "同组跨子集", "泄漏",
         "同一视频/序列的帧被拆到不同子集", rule_leak_group_across_splits),

    Rule("label_out_of_bounds", "标注越界", "标签",
         "框超出图像范围，训练会学到错误位置", rule_label_out_of_bounds),
    Rule("label_invalid_size", "零面积框", "标签",
         "宽或高为 0，无法构成有效目标", rule_label_invalid_size),
    Rule("label_box_area", "框面积异常", "标签",
         "占满整图或小到可疑的框", rule_label_box_area),
    Rule("label_duplicate", "重复标注", "标签",
         "同图内类别与坐标完全相同的框", rule_label_duplicate),
    Rule("image_without_annotation", "无标注图", "标签",
         "检测任务可作背景负样本，分类任务中无效", rule_image_without_annotation),
    Rule("label_orphan", "孤儿标签", "标签",
         "有标签文件但找不到对应图像", rule_orphan_labels),

    Rule("category_name_variants", "类别名变体", "类别",
         "疑似同义但写法不同的类别名，应合并", rule_category_name_variants),
    Rule("class_too_few", "类别样本过少", "类别",
         "实例数过少，训练效果差且指标波动大", rule_class_too_few),
]

RULES_BY_ID: Dict[str, Rule] = {r.id: r for r in RULES}
