"""清洗引擎：按规则检查，并按需执行处置。

    report = clean(bundle, CleanConfig())            # dry-run，只看问题
    report = clean(bundle, CleanConfig(), apply=True) # 确认后落盘

dry-run 与正式清洗走同一套判定逻辑，所以「预览到什么，执行时就会做什么」。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional

from ..ir import BBox, DatasetBundle
from .findings import (
    ACTION_CLAMP,
    ACTION_ORDER,
    ACTION_REMOVE_ANNOTATION,
    ACTION_REMOVE_IMAGE,
    SEVERITY_ORDER,
    CleanReport,
    Finding,
)
from .rules import RULES, RULES_BY_ID, CleanConfig, CleanContext


def list_rules() -> List[Dict[str, str]]:
    """列出所有规则，供前端展示与选择性关闭。"""
    return [
        {
            "id": r.id,
            "title": r.title,
            "category": r.category,
            "description": r.description,
        }
        for r in RULES
    ]


def clean(
    bundle: DatasetBundle,
    config: Optional[CleanConfig] = None,
    apply: bool = False,
) -> CleanReport:
    """执行清洗检查；apply=True 时按处置建议修改 bundle。"""
    config = config or CleanConfig()
    started = time.perf_counter()

    report = CleanReport(
        dry_run=not apply,
        images_before=len(bundle.images),
        annotations_before=len(bundle.annotations),
    )
    ctx = CleanContext(bundle, config, report)

    disabled = set(config.disabled_rules or ())

    for rule in RULES:
        if rule.id in disabled:
            report.skipped_rules.append(rule.id)
            continue
        try:
            findings = rule.check(ctx) or []
        except Exception as exc:  # 单条规则失败不应让整次清洗崩掉
            report.warnings.append(f"规则 {rule.id} 执行失败: {exc}")
            continue

        report.rules_run.append(rule.id)
        report.findings.extend(_cap(findings, rule.id, config.max_findings_per_rule, report))

    _summarize(report)

    if apply:
        _apply(bundle, report)

    report.images_after = len(bundle.images)
    report.annotations_after = len(bundle.annotations)
    report.duration_sec = time.perf_counter() - started
    return report


# ---------------------------------------------------------------------------


def _cap(
    findings: List[Finding],
    rule_id: str,
    limit: int,
    report: CleanReport,
) -> List[Finding]:
    """限制单条规则的明细数量，避免百万级明细撑爆响应。"""
    if limit <= 0 or len(findings) <= limit:
        return findings
    kept = findings[:limit]
    kept.append(
        Finding(
            rule=rule_id,
            severity=findings[0].severity,
            message=f"该规则共发现 {len(findings)} 条问题，明细仅展示前 {limit} 条",
            action=findings[0].action,
            detail={"total": len(findings), "truncated": True},
        )
    )
    return kept


def _summarize(report: CleanReport) -> None:
    by_rule: Counter = Counter(f.rule for f in report.findings)
    by_severity: Counter = Counter(f.severity for f in report.findings)
    report.counts_by_rule = dict(by_rule.most_common())
    report.counts_by_severity = {
        s: by_severity.get(s, 0) for s in SEVERITY_ORDER
    }


def _apply(bundle: DatasetBundle, report: CleanReport) -> None:
    """按 ACTION_ORDER 执行处置。

    顺序有讲究：先修（裁剪）再删标注、最后删图，
    这样「先裁剪后发现变成无效框」的情况能被同一轮处理掉。
    """
    applied: Counter = Counter()

    # 1) 裁剪越界框
    for f in report.findings:
        if f.action != ACTION_CLAMP or f.annotation is None or f.image is None:
            continue
        if _clamp(f.annotation, f.image):
            applied[ACTION_CLAMP] += 1
        else:
            # 完全在图像之外，裁剪后没有面积 -> 退化为删除标注
            if bundle.remove_annotation(f.annotation):
                applied[ACTION_REMOVE_ANNOTATION] += 1

    # 2) 删除单个标注
    for f in report.findings:
        if f.action != ACTION_REMOVE_ANNOTATION or f.annotation is None:
            continue
        if bundle.remove_annotation(f.annotation):
            applied[ACTION_REMOVE_ANNOTATION] += 1

    # 3) 删除整图（同一张图可能被多条规则命中，只删一次）
    removed_uids = set()
    for f in report.findings:
        if f.action != ACTION_REMOVE_IMAGE or not f.image_uid:
            continue
        if f.image_uid in removed_uids:
            continue
        removed_uids.add(f.image_uid)
        if bundle.remove_image(f.image_uid):
            applied[ACTION_REMOVE_IMAGE] += 1

    report.actions_applied = {a: applied.get(a, 0) for a in ACTION_ORDER if applied.get(a)}

    # 删除标注后可能产生新的「空标注图」，必须提醒：
    # 检测任务里这是合法背景图，分类任务里则是无效样本。
    emptied = _count_annotationless(bundle)
    if emptied:
        report.warnings.append(
            f"删除标注后有 {emptied} 张图变成无标注状态。"
            "检测任务中可作背景负样本保留；分类任务中应删除。"
        )


def _clamp(annotation, image) -> bool:
    """把框裁剪回图像范围内。裁剪后无面积则返回 False。"""
    bbox = annotation.bbox
    if bbox is None:
        return False

    w, h = float(image.width), float(image.height)
    if w <= 0 or h <= 0:
        return False

    x1 = min(max(bbox.x1, 0.0), w)
    y1 = min(max(bbox.y1, 0.0), h)
    x2 = min(max(bbox.x2, 0.0), w)
    y2 = min(max(bbox.y2, 0.0), h)

    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return False

    annotation.bbox = BBox(x1, y1, x2, y2)
    return True


def _count_annotationless(bundle: DatasetBundle) -> int:
    index = bundle.annotation_index()
    return sum(1 for uid in bundle.images if not index.get(uid))
