"""API 路由汇总。"""

from __future__ import annotations

from fastapi import APIRouter

from .routes import datasets, deploy, eval, files, infer, models, prelabel, system, train

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(datasets.router)
api_router.include_router(train.router)
api_router.include_router(eval.router)
api_router.include_router(models.router)
api_router.include_router(deploy.router)
api_router.include_router(infer.router)
api_router.include_router(prelabel.router)
api_router.include_router(files.router)

__all__ = ["api_router"]
