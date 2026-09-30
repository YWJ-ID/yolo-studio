"""清洗结果的数据结构。

设计原则（见 README）：
    * 每条规则产出「问题清单 + 处置建议」，不直接删；
    * `apply=False` 时纯报告（dry-run），人工确认后才落盘；
    * 任何处置都可回溯——Finding 里保留了具体是哪个规则、哪张图、哪个框。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir import Annotation, ImageRecord

# 严重程度
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

SEVERITY_ORDER = (SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO)

# 处置方式
ACTION_REPORT = "report"                      # 仅报告，不自动处置
ACTION_REMOVE_IMAGE = "remove_image"          # 删除整张图（连同其标注）
ACTION_REMOVE_ANNOTATION = "remove_annotation"  # 删除单个标注框
ACTION_CLAMP = "clamp"                        # 把越界框裁剪回图像范围

ACTION_ORDER = (
    ACTION_CLAMP,
    ACTION_REMOVE_ANNOTATION,
    ACTION_REMOVE_IMAGE,
)


@dataclass
class Finding:
    """一条清洗发现。"""

    rule: str
    severity: str
    message: str
    action: str = ACTION_REPORT
    image_uid: Optional[str] = None
    image_path: Optional[str] = None
    category: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    # 内部引用，用于 apply 时直接操作对象；不参与序列化
    image: Optional[ImageRecord] = field(default=None, repr=False, compare=False)
    annotation: Optional[Annotation] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "action": self.action,
            "image_uid": self.image_uid,
            "image_path": self.image_path,
            "category": self.category,
            "detail": self.detail,
        }


@dataclass
class CleanReport:
    """一次清洗的完整结果。"""

    dry_run: bool = True
    findings: List[Finding] = field(default_factory=list)
    counts_by_rule: Dict[str, int] = field(default_factory=dict)
    counts_by_severity: Dict[str, int] = field(default_factory=dict)
    actions_applied: Dict[str, int] = field(default_factory=dict)
    images_before: int = 0
    images_after: int = 0
    annotations_before: int = 0
    annotations_after: int = 0
    hashed_sha1: int = 0
    hashed_phash: int = 0
    near_duplicate_pairs: int = 0
    rules_run: List[str] = field(default_factory=list)
    skipped_rules: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def images_removed(self) -> int:
        return max(0, self.images_before - self.images_after)

    @property
    def annotations_removed(self) -> int:
        return max(0, self.annotations_before - self.annotations_after)

    def summary(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "images_before": self.images_before,
            "images_after": self.images_after,
            "annotations_before": self.annotations_before,
            "annotations_after": self.annotations_after,
            "images_removed": self.images_removed,
            "annotations_removed": self.annotations_removed,
            "counts_by_severity": self.counts_by_severity,
            "counts_by_rule": self.counts_by_rule,
            "actions_applied": self.actions_applied,
            "hashed_sha1": self.hashed_sha1,
            "hashed_phash": self.hashed_phash,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "rules_run": self.rules_run,
            "skipped_rules": self.skipped_rules,
            "duration_sec": round(self.duration_sec, 2),
            "warnings": self.warnings,
        }

    def to_dict(self, max_findings: int = 500) -> Dict[str, Any]:
        """序列化。findings 可能上万条，默认截断以免响应过大。"""
        return {
            **self.summary(),
            "findings_total": len(self.findings),
            "findings": [f.to_dict() for f in self.findings[:max_findings]],
        }
