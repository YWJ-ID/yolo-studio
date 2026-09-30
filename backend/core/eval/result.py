"""评估结果的数据结构与归一化。

`build_result()` 是纯函数：输入是子进程抽出的**原始 payload**（numpy 已转普通类型），
输出统一的 `EvalResult`。这样"结果长什么样"与"怎么从 ultralytics 拿出来"解耦，
前者可以脱离 ultralytics 单独测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .spec import RESULT_FILE

# 混淆矩阵方向：行=真实标签，列=预测标签（与 ultralytics 绘图一致）
CM_AXIS = "rows=真实,cols=预测"
BACKGROUND_LABEL = "background"


@dataclass
class ClassMetrics:
    """单个类别的指标。"""

    index: int
    name: str
    instances: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "instances": self.instances,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ap50": self.ap50,
            "ap50_95": self.ap50_95,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassMetrics":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EvalResult:
    """一次评估的完整结果。"""

    ok: bool = False
    error: str = ""
    split: str = ""
    task: str = "detect"
    weights: str = ""
    data_yaml: str = ""
    model_name: str = ""
    job_id: str = ""
    tag: str = ""
    eval_id: str = ""
    created_at: str = ""
    duration_sec: float = 0.0

    # 总体指标（键名与前端约定：precision / recall / f1 / mAP50 / mAP50-95 / fitness）
    overall: Dict[str, float] = field(default_factory=dict)
    per_class: List[ClassMetrics] = field(default_factory=list)
    # 每张图的耗时（毫秒）
    speed: Dict[str, float] = field(default_factory=dict)
    # 混淆矩阵（含 background 行/列）
    confusion_matrix: Optional[Dict[str, Any]] = None
    # 过程图像文件名（相对评估目录）
    artifacts: List[str] = field(default_factory=list)

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "split": self.split,
            "task": self.task,
            "weights": self.weights,
            "data_yaml": self.data_yaml,
            "model_name": self.model_name,
            "job_id": self.job_id,
            "tag": self.tag,
            "eval_id": self.eval_id,
            "created_at": self.created_at,
            "duration_sec": self.duration_sec,
            "overall": self.overall,
            "per_class": [c.to_dict() for c in self.per_class],
            "speed": self.speed,
            "confusion_matrix": self.confusion_matrix,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        payload["per_class"] = [ClassMetrics.from_dict(c) for c in d.get("per_class", [])]
        return cls(**payload)

    def save(self, path=None) -> Path:
        p = Path(path) if path else None
        if p is None:
            raise ValueError("保存评估结果需要给出路径")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path) -> "EvalResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------- 便捷查询（供对比页使用） ----------

    def metric(self, key: str) -> Optional[float]:
        if key in self.overall:
            return self.overall[key]
        return None

    def headline(self) -> Dict[str, float]:
        keys = ("precision", "recall", "f1", "mAP50", "mAP50-95")
        return {k: self.overall[k] for k in keys if k in self.overall}


def _num(value: Any, default: float = 0.0) -> float:
    """把可能是 numpy 标量/字符串的值转成 float。"""
    v = _optional_num(value)
    return default if v is None else v


def _optional_num(value: Any) -> Optional[float]:
    """转成 float；无法解析或 NaN 时返回 None。

    返回 None 而不是 0：区分「没有这个指标」与「指标真的是 0」很重要——
    把缺失指标显示成 0 会被误读为「效果很差」。
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round(v, 6)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _overall_from_results_dict(results_dict: Dict[str, Any]) -> Dict[str, float]:
    """从 ultralytics 的 results_dict 取出常用指标。

    键名与训练指标保持一致（前端同一条曲线/表格就能复用）。
    解析不出来的指标直接不写入，不补 0。
    """
    out: Dict[str, float] = {}
    mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
        "fitness": "fitness",
        # 分类任务
        "metrics/accuracy_top1": "top1",
        "metrics/accuracy_top5": "top5",
    }
    for key, value in (results_dict or {}).items():
        dst = mapping.get(str(key))
        if not dst:
            continue
        num = _optional_num(value)
        if num is not None:
            out[dst] = num
    return out


def build_result(payload: Dict[str, Any], spec, eval_id: str, duration_sec: float = 0.0) -> EvalResult:
    """把子进程抽出的原始 payload 归一为 EvalResult（纯函数）。"""
    result = EvalResult(
        ok=bool(payload.get("ok", True)),
        error=str(payload.get("error", "") or ""),
        split=spec.split,
        task=getattr(spec, "task", "detect"),
        weights=str(getattr(spec, "weights", "")),
        data_yaml=str(getattr(spec, "data_yaml", "")),
        job_id=str(getattr(spec, "job_id", "") or ""),
        tag=str(getattr(spec, "tag", "") or ""),
        eval_id=eval_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        duration_sec=round(float(duration_sec or 0.0), 3),
        model_name=Path(str(getattr(spec, "weights", ""))).name,
    )

    result.overall = _overall_from_results_dict(payload.get("results_dict", {}))

    classes: List[ClassMetrics] = []
    for item in payload.get("per_class", []) or []:
        classes.append(
            ClassMetrics(
                index=_int(item.get("index")),
                name=str(item.get("name", "")),
                instances=_int(item.get("instances")),
                precision=_num(item.get("precision")),
                recall=_num(item.get("recall")),
                f1=_num(item.get("f1")),
                ap50=_num(item.get("ap50")),
                ap50_95=_num(item.get("ap50_95")),
            )
        )
    result.per_class = classes

    # 类别平均 F1：ultralytics 的 results_dict 里没有 F1，按各类别 F1 求均值补上
    if "f1" not in result.overall and classes:
        vals = [c.f1 for c in classes]
        if vals:
            result.overall["f1"] = round(sum(vals) / len(vals), 6)

    result.speed = {str(k): _num(v) for k, v in (payload.get("speed") or {}).items()}

    cm = payload.get("confusion_matrix")
    if cm and cm.get("matrix"):
        result.confusion_matrix = {
            "axis": CM_AXIS,
            "labels": list(cm.get("labels") or []),
            "matrix": [[_int(v) for v in row] for row in cm["matrix"]],
        }

    result.artifacts = [str(name) for name in (payload.get("artifacts") or [])]
    return result


def load_result(run_dir) -> Optional[EvalResult]:
    """读取评估目录里的结果文件；不存在返回 None。"""
    path = Path(run_dir) / RESULT_FILE
    if not path.is_file():
        return None
    try:
        return EvalResult.load(path)
    except Exception:
        return None
