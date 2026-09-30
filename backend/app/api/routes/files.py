"""本地文件读取接口（仅用于预览磁盘上的数据集图片）。

安全说明：
    默认配置（allowed_roots 为空）不限制路径，**仅适用于本机单机开发**。
    若要对外提供服务，必须设置 YOLO_STUDIO_ALLOWED_ROOTS 白名单。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from core.io import is_image

from ...config import settings

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/image")
def get_image(path: str = Query(..., description="图像绝对路径")):
    """按绝对路径返回图像文件，供前端叠加标注框预览。"""
    if not settings.is_path_allowed(path):
        raise HTTPException(status_code=403, detail="路径不在允许的白名单内")

    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    if not is_image(p):
        raise HTTPException(status_code=400, detail="不是受支持的图像文件")

    return FileResponse(p)


@router.get("/download")
def download(path: str = Query(..., description="文件绝对路径")):
    """下载生成的产物（如 HTML 质量报告）。

    仅允许下载存储目录下的文件，避免这个接口变成任意文件读取。
    """
    if not settings.is_path_allowed(path):
        raise HTTPException(status_code=403, detail="路径不在允许的白名单内")

    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")

    storage = settings.storage_dir.resolve()
    try:
        p.resolve().relative_to(storage)
    except ValueError:
        raise HTTPException(status_code=403, detail="只能下载存储目录内的产物") from None

    return FileResponse(p, filename=p.name)
