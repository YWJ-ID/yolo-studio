"""YOLO Studio 后端入口。

开发模式：
    cd backend && uvicorn app.main:app --reload --port 8000

生产模式（前端已 build 到 frontend/dist）：
    FastAPI 会直接托管静态文件，单端口访问。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.router import api_router
from .config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="YOLO 训练全流程可视化工作台",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
def _startup() -> None:
    settings.ensure_dirs()


@app.on_event("shutdown")
def _shutdown() -> None:
    # 不结束训练/评估子进程，只停监控线程；新进程启动时会接管训练任务
    from .services import shutdown_managers

    shutdown_managers()


# ---------- 静态前端托管（生产模式） ----------
# 开发时前端跑在 Vite (5173)，这里的挂载无副作用。
_dist = Path(settings.frontend_dist)
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        # API 路径绝不落到 SPA 兜底，否则前端会把 HTML 当成 JSON 解析
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi"):
            raise HTTPException(status_code=404, detail=f"接口不存在: /{full_path}")
        candidate = _dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
