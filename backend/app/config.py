"""后端全局配置。

所有路径以环境变量为准，默认值适配本机开发环境。
环境变量前缀：YOLO_STUDIO_
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# 项目根 = backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).resolve() if raw else default


def _env_list(key: str, default: list) -> list:
    """读一个「分号或逗号分隔」的列表环境变量；未设置时用默认值。"""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in re.split(r"[;,]", raw) if item.strip()]


class Settings:
    """轻量配置对象（避免为骨架引入 pydantic-settings 的强耦合）。"""

    app_name: str = "YOLO Studio"
    version: str = "0.1.0"

    # 存储根目录：sqlite、数据集版本、训练产物
    storage_dir: Path = _env_path("YOLO_STUDIO_STORAGE", BACKEND_DIR / "storage")
    # 生成的数据集版本
    datasets_dir: Path = _env_path("YOLO_STUDIO_DATASETS", BACKEND_DIR / "storage" / "datasets")
    # 训练产物 runs/
    runs_dir: Path = _env_path("YOLO_STUDIO_RUNS", BACKEND_DIR / "storage" / "runs")
    # 预训练权重缓存（把 .pt 放在这里，训练时按文件名引用即可，不必写绝对路径）
    weights_dir: Path = _env_path("YOLO_STUDIO_WEIGHTS", BACKEND_DIR / "storage" / "weights")
    # 评估产物（val / test 指标与图表）
    evals_dir: Path = _env_path("YOLO_STUDIO_EVALS", BACKEND_DIR / "storage" / "evals")
    # 模型库（每个模型目录一份 model_card.json）
    models_dir: Path = _env_path("YOLO_STUDIO_MODELS", BACKEND_DIR / "storage" / "models")
    # 导出产物（ONNX / TorchScript 等）
    deploys_dir: Path = _env_path("YOLO_STUDIO_DEPLOYS", BACKEND_DIR / "storage" / "deploys")
    # 上传/导入的原始数据
    uploads_dir: Path = _env_path("YOLO_STUDIO_UPLOADS", BACKEND_DIR / "storage" / "uploads")

    db_path: Path = _env_path("YOLO_STUDIO_DB", BACKEND_DIR / "storage" / "studio.db")

    # 训练使用的 Python 解释器，默认当前解释器。
    # 需要跑在别的环境时用 YOLO_STUDIO_PYTHON 覆盖。
    python_executable: Path = Path(
        os.environ.get("YOLO_STUDIO_PYTHON") or sys.executable
    ).resolve()

    # 训练默认设备：auto = 有 CUDA 用 GPU，否则 CPU
    train_device: str = os.environ.get("YOLO_STUDIO_DEVICE", "auto")

    # API 监听端口。避开 8000（本机已被 OpenAI 兼容网关占用）
    api_port: int = int(os.environ.get("YOLO_STUDIO_PORT", "8010"))

    # CORS 允许的来源。用 YOLO_STUDIO_CORS_ORIGINS 覆盖（分号或逗号分隔），
    # 例如把服务放到别的机器上、从 http://192.168.1.20:8010 访问时需要加上该来源。
    # 同源访问（后端托管前端，单端口）不受 CORS 限制，无需配置。
    cors_origins: list = _env_list(
        "YOLO_STUDIO_CORS_ORIGINS",
        ["http://localhost:5173", "http://127.0.0.1:5173"],
    )

    # 前端构建产物（生产模式下由 FastAPI 托管）
    frontend_dist: Path = PROJECT_DIR / "frontend" / "dist"

    # 图片读取白名单。
    # 默认空列表 = 不限制（仅适用于本机单机开发）。
    # 若要把服务暴露到局域网/公网，必须通过 YOLO_STUDIO_ALLOWED_ROOTS 指定允许的根目录，
    # 用分号分隔，例如：D:\dataset;D:\phpstudy_pro\WWW\yolo-studio\backend\storage
    allowed_roots: list = [
        Path(p).resolve()
        for p in os.environ.get("YOLO_STUDIO_ALLOWED_ROOTS", "").split(";")
        if p.strip()
    ]

    # 质量报告输出目录
    reports_dir: Path = _env_path("YOLO_STUDIO_REPORTS", BACKEND_DIR / "storage" / "reports")

    def ensure_dirs(self) -> None:
        for d in (
            self.storage_dir,
            self.datasets_dir,
            self.runs_dir,
            self.uploads_dir,
            self.reports_dir,
            self.weights_dir,
            self.evals_dir,
            self.models_dir,
            self.deploys_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def is_path_allowed(self, path) -> bool:
        """校验本地文件读取是否在白名单内。白名单为空时不做限制。"""
        if not self.allowed_roots:
            return True
        try:
            resolved = Path(path).resolve()
        except Exception:
            return False
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False


settings = Settings()
