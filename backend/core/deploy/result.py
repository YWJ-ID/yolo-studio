"""导出结果的记录与完整性校验（M4-01 / M4-03）。

每个产物都记下路径、大小与 sha256：
  * 大小与哈希在导出时算一次，作为「当时是什么样」的基线；
  * `verify_result()` 事后重新计算并比对，用于发现文件被截断/替换/删除。

不记录「模型准确率」之类需要重新推理才有意义的指标——那是评估模块（M3）的职责。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .spec import RESULT_FILE

# 分块哈希，避免把大文件一次性读进内存
_CHUNK = 1024 * 1024


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path) -> int:
    """目录型产物（如 openvino）的总大小。"""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


@dataclass
class Artifact:
    """一个导出产物。"""

    format: str
    ok: bool = False
    path: str = ""            # 导出目录内的路径（copy_artifacts 后）
    source_path: str = ""     # ultralytics 实际写出的路径
    is_dir: bool = False
    size: int = 0
    sha256: str = ""
    duration_sec: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "ok": self.ok,
            "path": self.path,
            "source_path": self.source_path,
            "is_dir": self.is_dir,
            "size": self.size,
            "sha256": self.sha256,
            "duration_sec": round(self.duration_sec, 3),
            "error": self.error,
            "name": Path(self.path).name if self.path else "",
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DeployResult:
    """一次导出的完整结果。"""

    ok: bool = False
    error: str = ""
    weights: str = ""
    model_name: str = ""
    task: str = "detect"
    job_id: str = ""
    deploy_id: str = ""
    tag: str = ""
    imgsz: int = 640
    device: str = "cpu"
    requested: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    created_at: str = ""
    duration_sec: float = 0.0

    def succeeded(self) -> List[Artifact]:
        return [a for a in self.artifacts if a.ok]

    def failed(self) -> List[Artifact]:
        return [a for a in self.artifacts if not a.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "weights": self.weights,
            "model_name": self.model_name,
            "task": self.task,
            "job_id": self.job_id,
            "deploy_id": self.deploy_id,
            "tag": self.tag,
            "imgsz": self.imgsz,
            "device": self.device,
            "requested": self.requested,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "created_at": self.created_at,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeployResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        payload["artifacts"] = [Artifact.from_dict(a) for a in d.get("artifacts", [])]
        return cls(**payload)

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path) -> "DeployResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def describe_artifact(path, copy_artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """算出产物的大小与哈希；目录型产物累加大小、哈希记空。

    copy_artifacts_dir 给定时把**文件型**产物复制过去，返回复制后的信息。
    目录型产物不复制（可能很大且文件众多），只记录原路径。
    """
    src = Path(path)
    is_dir = src.is_dir()
    info: Dict[str, Any] = {
        "source_path": str(src),
        "is_dir": is_dir,
        "path": str(src),
        "size": dir_size(src) if is_dir else (src.stat().st_size if src.is_file() else 0),
        "sha256": "" if is_dir else (sha256_file(src) if src.is_file() else ""),
    }
    if copy_artifacts_dir is not None and src.is_file():
        dst_dir = Path(copy_artifacts_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            import shutil

            shutil.copy2(src, dst)
        info["path"] = str(dst)
        info["size"] = dst.stat().st_size
        info["sha256"] = sha256_file(dst)
    return info


@dataclass
class VerifyReport:
    """产物完整性校验结果（M4-03）。"""

    ok: bool = True
    checked: int = 0
    problems: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checked": self.checked, "problems": self.problems}


def verify_result(result: DeployResult) -> VerifyReport:
    """重新计算产物的大小与 sha256，与导出时记录的基线比对。"""
    report = VerifyReport()
    for a in result.artifacts:
        if not a.ok:
            continue  # 导出时就失败了，不算完整性问题
        report.checked += 1
        path = Path(a.path)
        if not path.exists():
            report.problems.append({"format": a.format, "path": a.path, "issue": "文件不存在"})
            continue
        if a.is_dir:
            actual = dir_size(path)
            if a.size and actual != a.size:
                report.problems.append(
                    {"format": a.format, "path": a.path, "issue": "目录大小不一致",
                     "expected": a.size, "actual": actual}
                )
            continue
        actual = path.stat().st_size
        if a.size and actual != a.size:
            report.problems.append(
                {"format": a.format, "path": a.path, "issue": "大小不一致",
                 "expected": a.size, "actual": actual}
            )
            continue
        if a.sha256:
            digest = sha256_file(path)
            if digest != a.sha256:
                report.problems.append(
                    {"format": a.format, "path": a.path, "issue": "内容已变化（sha256 不一致）"}
                )
    report.ok = not report.problems
    return report


def load_result(run_dir) -> Optional[DeployResult]:
    path = Path(run_dir) / RESULT_FILE
    if not path.is_file():
        return None
    try:
        return DeployResult.load(path)
    except Exception:
        return None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
