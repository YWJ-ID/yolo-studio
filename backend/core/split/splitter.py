"""数据集划分。

三种划分能力，可组合：

1. **分组划分（防泄漏）**：同一 `group` 的图像必须整体落入同一个子集。
   视频抽帧场景下，相邻帧几乎完全相同，一旦跨 train/val 指标就会虚高——
   这是最隐蔽也最致命的错误，因此**只要图像带 group 就默认强制分组**。

2. **分层划分**：按类别分层，保证长尾类在每个子集中都有样本。
   每张图（或每个组）以其**全局最稀有类别**作为分层标签，优先保障稀有类分布。

3. **尊重已有划分**：输入数据集自带 train/val/test 时默认保留，只对未划分部分进行划分。

所有随机性都由固定 seed 控制，可复现。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..ir import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, DatasetBundle, ImageRecord

UNLABELED = "__unlabeled__"

STRATEGY_RANDOM = "random"
STRATEGY_STRATIFIED = "stratified"

_SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


@dataclass
class SplitConfig:
    """划分参数。"""

    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42
    strategy: str = STRATEGY_STRATIFIED   # random | stratified
    respect_groups: bool = True           # 有 group 时强制整组同去
    respect_existing: bool = True         # 保留输入里已有的划分
    min_val: int = 1                      # val 至少放几个单元（够评估）
    min_test: int = 1

    def normalized_ratios(self) -> Tuple[float, float, float]:
        total = sum(self.ratios)
        if total <= 0:
            raise ValueError(f"ratios 之和必须为正: {self.ratios}")
        r = tuple(x / total for x in self.ratios)
        if len(r) != 3:
            raise ValueError(f"ratios 必须是 3 个数 (train, val, test): {self.ratios}")
        return r  # type: ignore[return-value]


@dataclass
class SplitReport:
    """划分结果摘要，用于前端展示与落盘记录。"""

    units_total: int = 0
    units_grouped: int = 0
    images_total: int = 0
    images_already_split: int = 0
    images_assigned: int = 0
    split_images: Dict[str, int] = field(default_factory=dict)
    split_units: Dict[str, int] = field(default_factory=dict)
    class_by_split: Dict[str, Dict[str, int]] = field(default_factory=dict)
    """val/test 中缺失的类别。缺失会使该类别指标变为 NaN，必须让用户知道。"""
    classes_missing_in_split: Dict[str, List[str]] = field(default_factory=dict)
    target_ratios: Dict[str, float] = field(default_factory=dict)
    actual_ratios: Dict[str, float] = field(default_factory=dict)
    group_conflicts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units_total": self.units_total,
            "units_grouped": self.units_grouped,
            "images_total": self.images_total,
            "images_already_split": self.images_already_split,
            "images_assigned": self.images_assigned,
            "split_images": self.split_images,
            "split_units": self.split_units,
            "class_by_split": self.class_by_split,
            "classes_missing_in_split": self.classes_missing_in_split,
            "target_ratios": self.target_ratios,
            "actual_ratios": self.actual_ratios,
            "group_conflicts": self.group_conflicts,
            "warnings": self.warnings,
        }


def assign_splits(
    bundle: DatasetBundle,
    config: Optional[SplitConfig] = None,
) -> SplitReport:
    """就地写入每个 ImageRecord.split，并返回划分报告。"""
    config = config or SplitConfig()
    ratios = config.normalized_ratios()
    report = SplitReport()

    images = list(bundle.images.values())
    report.images_total = len(images)
    if not images:
        report.warnings.append("没有图像可划分")
        return report

    # ---------- 1. 已有划分的画像：保留 ----------
    locked: Dict[str, str] = {}
    if config.respect_existing:
        for im in images:
            if im.split in _SPLIT_ORDER:
                locked[im.uid] = im.split
        report.images_already_split = len(locked)

    # ---------- 2. 组装划分单元 ----------
    units, units_grouped = _build_units(images, locked, config, report)
    report.units_total = len(units)
    report.units_grouped = units_grouped

    pending = [u for u in units if u.assigned is None]
    if not pending:
        report.warnings.append("所有图像都已带划分，无需重新划分")
        _fill_report(bundle, report)
        return report

    # ---------- 3. 计算分层标签 ----------
    strat_of_unit = _stratify(pending, bundle, config)

    # ---------- 4. 按层分配 ----------
    rng = random.Random(config.seed)
    by_stratum: Dict[str, List[_Unit]] = defaultdict(list)
    for unit in pending:
        by_stratum[strat_of_unit[unit.key]].append(unit)

    for stratum in sorted(by_stratum):
        _allocate_weighted(by_stratum[stratum], ratios, bundle, rng)

    # ---------- 5. 全局修正最小样本量 ----------
    # 注意：必须在所有分层分配完成后做一次全局修正。
    # 若按分层各拉一个到 val，val 占比会被各层放大（100 个分层 -> val 变三分之一）。
    _ensure_minimums(pending, config, report)

    # ---------- 6. 写回 ----------
    for unit in units:
        target = unit.assigned
        if target is None:
            report.warnings.append(f"单元 {unit.key} 未获得划分，默认归入 train")
            target = SPLIT_TRAIN
        for uid in unit.uids:
            bundle.images[uid].split = target
            report.images_assigned += 1

    _fill_report(bundle, report)
    _check_ratio_deviation(report, ratios)
    return report


# ---------------------------------------------------------------------------
# 内部结构
# ---------------------------------------------------------------------------


@dataclass
class _Unit:
    """划分的最小单元：一张图，或一个必须整体移动的组。"""

    key: str
    uids: List[str]
    group: Optional[str] = None
    assigned: Optional[str] = None


def _build_units(
    images: List[ImageRecord],
    locked: Dict[str, str],
    config: SplitConfig,
    report: SplitReport,
) -> Tuple[List[_Unit], int]:
    """把图像组装成划分单元。

    有 group 且开启 respect_groups 时，同组图像合并为一个单元；
    若组内部分图像已带划分，整组跟随该划分（冲突则记录警告）。
    """
    use_groups = config.respect_groups and any(im.group for im in images)

    if not use_groups:
        units = []
        for im in images:
            unit = _Unit(key=im.uid, uids=[im.uid], group=None)
            if im.uid in locked:
                unit.assigned = locked[im.uid]
            units.append(unit)
        return units, 0

    buckets: Dict[str, List[ImageRecord]] = defaultdict(list)
    for im in images:
        buckets[im.group or im.uid].append(im)

    units: List[_Unit] = []
    grouped = 0
    for key in sorted(buckets):
        members = buckets[key]
        uids = [im.uid for im in members]
        unit = _Unit(key=key, uids=uids, group=members[0].group)

        if unit.group:
            grouped += 1

        existing = {locked[uid] for uid in uids if uid in locked}
        if len(existing) == 1:
            unit.assigned = existing.pop()
        elif len(existing) > 1:
            # 同组被分到了不同子集 —— 这正是要防的泄漏。
            # 统一时优先保留 train：它是数据量最大的子集，
            # 把帧并入 train 比并入 val/test 对训练更有利。
            chosen = next(
                (s for s in _SPLIT_ORDER if s in existing),
                sorted(existing)[0],
            )
            unit.assigned = chosen
            report.group_conflicts.append(
                f"组 '{key}' 已存在于多个子集 {sorted(existing)}，已统一为 '{chosen}'"
            )
        units.append(unit)

    return units, grouped


def _stratify(
    units: Sequence[_Unit],
    bundle: DatasetBundle,
    config: SplitConfig,
) -> Dict[str, str]:
    """为每个单元计算分层标签。

    分层标签取该单元内**全局最稀有**的类别：
    这样稀有类会被优先均匀摊到各子集，而不是被随机划分淹没。
    """
    if config.strategy == STRATEGY_RANDOM:
        return {u.key: "__all__" for u in units}

    global_counts: Counter = Counter(bundle.count_by_category())

    labels_of_unit: Dict[str, set] = {}
    for unit in units:
        cats = {
            a.category
            for uid in unit.uids
            for a in bundle.annotations_of(uid)
        }
        labels_of_unit[unit.key] = cats

    result: Dict[str, str] = {}
    for unit in units:
        cats = labels_of_unit[unit.key]
        if not cats:
            result[unit.key] = UNLABELED
            continue
        result[unit.key] = min(cats, key=lambda c: (global_counts.get(c, 0), c))
    return result


def _allocate_weighted(
    units: List[_Unit],
    ratios: Tuple[float, float, float],
    bundle: DatasetBundle,
    rng: random.Random,
) -> None:
    """按**图像数量**（而非单元个数）把单元分配到三个子集。

    为什么不能按个数分：一个「视频组」单元可能有 80 帧，而一张网图单元只有 1 帧，
    体积差两个数量级。按个数分会让 8:1:1 严重失真（实测偏移到 83.5:6.9:9.6）。

    做法是经典的「最大优先 + 最小缺口」贪心：
    单元按体积从大到小放入当前最缺额的子集，保证各子集体量贴近目标比例。

    随机性来源：同体积单元之间的顺序由 seed 决定的随机值打散。
    若不用随机值而是按 key 排序，整个划分会与 seed 无关（完全不随机）。
    """
    sizes = {unit.key: len(unit.uids) for unit in units}
    total = sum(sizes.values())
    if total <= 0:
        return

    # 同体积单元的排序随机化 —— 这是 seed 唯一能起作用的地方
    tiebreak = {unit.key: rng.random() for unit in units}

    targets = [total * r for r in ratios]
    fill = [0.0, 0.0, 0.0]

    # 体积大的先放：大单元放完，小单元用于填补尾差，碎片最少
    for unit in sorted(units, key=lambda u: (-sizes[u.key], tiebreak[u.key])):
        # 缺口最大者优先；平手时偏向 train（索引小的）
        idx = max(range(3), key=lambda j: (targets[j] - fill[j], -j))
        unit.assigned = _SPLIT_ORDER[idx]
        fill[idx] += sizes[unit.key]


def _allocate(n: int, ratios: Tuple[float, float, float]) -> Dict[str, int]:
    """把 n 个等体积单元按比例分配（保留给测试与兜底使用）。"""
    if n <= 0:
        return {s: 0 for s in _SPLIT_ORDER}

    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]

    remainder = n - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    return dict(zip(_SPLIT_ORDER, counts))


def _ensure_minimums(
    units: List[_Unit],
    config: SplitConfig,
    report: SplitReport,
) -> None:
    """全局修正：保证 val/test 至少各有 min_val / min_test 个单元。

    修正来源是 train，且优先从「train 单元最多的分层」里取，
    这样对分层均衡的破坏最小。仅在总量允许时才修正。
    """
    mins = {SPLIT_VAL: config.min_val, SPLIT_TEST: config.min_test}
    if not any(mins.values()):
        return

    def count(split: str) -> int:
        return sum(1 for u in units if u.assigned == split)

    for target_split in (SPLIT_VAL, SPLIT_TEST):
        minimum = mins[target_split]
        while count(target_split) < minimum and count(SPLIT_TRAIN) > 1:
            # 找出 train 单元最多的分层，从里面挪一个最靠后的单元
            stratum_train: Dict[str, List[_Unit]] = defaultdict(list)
            for u in units:
                if u.assigned == SPLIT_TRAIN:
                    stratum_train[u.key].append(u)
            if not stratum_train:
                break
            biggest = max(stratum_train.values(), key=len)
            moved = biggest[-1]
            moved.assigned = target_split
            report.warnings.append(
                f"为满足 {target_split} 最小样本量，将单元 '{moved.key}' 从 train 移入 {target_split}"
            )

    if count(SPLIT_VAL) == 0 and count(SPLIT_TEST) == 0:
        report.warnings.append(
            "划分单元过少，无法同时保证 train/val/test 非空。"
            "若数据来自单个视频，这是防泄漏的必然结果——需要更多视频才能做有效验证。"
        )


def _check_ratio_deviation(
    report: SplitReport, target: Tuple[float, float, float]
) -> None:
    """记录实际比例，并在偏离目标较大时告警。

    比例失真必须让用户知道——他会以为 val 占 10%，实际可能只有 7%，
    这直接影响他对验证结果可信度的判断。
    """
    report.target_ratios = {s: round(r, 4) for s, r in zip(_SPLIT_ORDER, target)}
    total = sum(report.split_images.values())
    if not total:
        report.actual_ratios = {s: 0.0 for s in _SPLIT_ORDER}
        return

    report.actual_ratios = {
        s: round(report.split_images.get(s, 0) / total, 4) for s in _SPLIT_ORDER
    }

    worst = max(
        abs(report.actual_ratios[s] - r) for s, r in zip(_SPLIT_ORDER, target)
    )
    if worst > 0.03:
        detail = ", ".join(
            f"{s} {report.actual_ratios[s]:.1%}(目标 {r:.0%})"
            for s, r in zip(_SPLIT_ORDER, target)
        )
        report.warnings.append(
            f"实际划分比例与目标有偏差（最大 {worst:.1%}）: {detail}。"
            "样本按分组整体移动且各组体量差异大时会出现这种情况。"
        )


def _fill_report(bundle: DatasetBundle, report: SplitReport) -> None:
    """统计各子集的图像数与类别分布，并标出 val/test 中缺失的类别。"""
    report.split_images = {s: 0 for s in _SPLIT_ORDER}
    report.class_by_split = {s: {} for s in _SPLIT_ORDER}

    for im in bundle.images.values():
        split = im.split if im.split in _SPLIT_ORDER else None
        if split is None:
            continue
        report.split_images[split] += 1
        for ann in bundle.annotations_of(im.uid):
            bucket = report.class_by_split[split]
            bucket[ann.category] = bucket.get(ann.category, 0) + 1

    # 类别在验证/测试集缺失时该类别指标为 NaN。
    # 这里只报告、不强行把稀有类塞进各子集——那样会破坏既定比例，
    # 且小样本下"既要比例精确又要每类都有"本就无解，选择权应交回用户。
    all_classes = bundle.category_names()
    for split in (SPLIT_VAL, SPLIT_TEST):
        present = set(report.class_by_split.get(split, {}))
        missing = [c for c in all_classes if c not in present]
        if missing:
            report.classes_missing_in_split[split] = missing
            report.warnings.append(
                f"{split} 子集中缺少 {len(missing)} 个类别（这些类别的指标将为 NaN）: "
                + ", ".join(missing[:8])
                + (" …" if len(missing) > 8 else "")
            )
