"""多模型 / 多实验对比（M3-04）。

对比的前提是「可比」：不同划分（val vs test）的数字不能放在一张表里相互比较。
因此这里显式检查划分是否一致，不一致时给出提示而不是假装可比。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .result import EvalResult

# 展示顺序：先总体、后召回/精确，最后 AP
METRIC_ORDER = ("mAP50-95", "mAP50", "precision", "recall", "f1", "top1", "top5")

METRIC_LABELS = {
    "mAP50-95": "mAP50-95",
    "mAP50": "mAP50",
    "precision": "precision",
    "recall": "recall",
    "f1": "F1",
    "top1": "top1",
    "top5": "top5",
    "fitness": "fitness",
}


def compare_eval_results(
    items: Sequence[Tuple[str, EvalResult]],
    labels: Sequence[str] = ("mAP50-95", "mAP50"),
) -> Dict[str, Any]:
    """把多份评估结果整理成对比表。

    items: [(显示名, 评估结果), ...]
    labels: 除默认指标外还想额外展示的指标键（如训练过程最优值）

    返回 rows / class_table / metrics / notes，可直接喂给前端表格。
    """
    notes: List[str] = []
    usable: List[Tuple[str, EvalResult]] = [(name, r) for name, r in items if r is not None]

    if not usable:
        return {
            "rows": [],
            "class_table": [],
            "metrics": [],
            "splits": [],
            "notes": ["没有可对比的评估结果"],
        }

    splits = sorted({r.split for _, r in usable if r.split})
    if len(splits) > 1:
        notes.append(
            "参与对比的评估使用了不同划分（"
            + "、".join(splits)
            + "），这些数字之间不具备可比性，请先在相同划分上评估后再比较。"
        )

    # 指标列：所有结果里出现过的键，按预设顺序，其余追加在后面
    present: List[str] = []
    for _, r in usable:
        for key in r.overall:
            if key not in present:
                present.append(key)
    metrics = [m for m in METRIC_ORDER if m in present]
    metrics += [m for m in present if m not in metrics]
    metrics += [m for m in labels if m in present and m not in metrics]

    rows: List[Dict[str, Any]] = []
    for name, r in usable:
        values = {m: r.overall.get(m) for m in metrics if m in r.overall}
        rows.append(
            {
                "label": name,
                "model_name": r.model_name,
                "eval_id": r.eval_id,
                "split": r.split,
                "task": r.task,
                "job_id": r.job_id,
                "num_classes": len(r.per_class),
                "values": values,
            }
        )

    # 逐类对比：以 AP50-95 为口径。缺失的类别留空（不补 0）
    class_names: List[str] = []
    for _, r in usable:
        for c in r.per_class:
            if c.name not in class_names:
                class_names.append(c.name)
    class_table: List[Dict[str, Any]] = []
    for cls in class_names:
        entry: Dict[str, Any] = {"name": cls, "values": {}}
        for name, r in usable:
            match = next((c for c in r.per_class if c.name == cls), None)
            entry["values"][name] = None if match is None else match.ap50_95
        class_table.append(entry)

    # 只在类别集合不一致时提示，避免噪声
    class_sets = [tuple(sorted(c.name for c in r.per_class)) for _, r in usable]
    if len(set(class_sets)) > 1:
        notes.append("参与对比的模型类别集合不一致，逐类对比表中缺失项留空。")

    if any(not r.ok for _, r in usable):
        failed = [name for name, r in usable if not r.ok]
        notes.append("以下评估未成功完成，其指标可能缺失：" + "、".join(failed))

    return {
        "rows": rows,
        "class_table": class_table,
        "metrics": metrics,
        "metric_labels": [METRIC_LABELS.get(m, m) for m in metrics],
        "splits": splits,
        "labels": [name for name, _ in usable],
        "notes": notes,
    }


def best_by(rows: Sequence[Dict[str, Any]], metric: str, mode: str = "max") -> Optional[str]:
    """在对比结果里挑出某指标最优的模型标签。用于界面高亮。"""
    candidates = [
        (str(row["label"]), row["values"][metric])
        for row in rows
        if metric in (row.get("values") or {}) and row["values"][metric] is not None
    ]
    if not candidates:
        return None
    pick = max if mode == "max" else min
    return pick(candidates, key=lambda x: x[1])[0]
