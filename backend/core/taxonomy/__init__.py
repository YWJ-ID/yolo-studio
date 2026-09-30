"""类别体系规范化模块。"""

from __future__ import annotations

from .taxonomy import (
    TaxonomyConfig,
    TaxonomyReport,
    apply_taxonomy,
    normalize_key,
    sanitize_class_name,
    suggest_merges,
    validate_class_names,
)

__all__ = [
    "TaxonomyConfig",
    "TaxonomyReport",
    "apply_taxonomy",
    "normalize_key",
    "sanitize_class_name",
    "suggest_merges",
    "validate_class_names",
]
