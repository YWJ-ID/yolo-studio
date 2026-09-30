"""图像与文件的基础工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageFile

# 允许读取被截断的图片，避免整批数据因个别坏图中断
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def is_image(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def probe_size(path) -> Optional[Tuple[int, int]]:
    """只读图像头获取 (width, height)，不解码像素，速度快。失败返回 None。"""
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None


def is_readable(path) -> bool:
    """完整校验图像可解码（比 probe_size 慢，清洗阶段用）。"""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def get_exif_orientation(path) -> Optional[int]:
    """返回 EXIF 方向值（1-8），无则 None。

    DMS 车载数据常见手机/相机拍摄，方向标记不一致会导致标注框错位。
    """
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            return exif.get(0x0112)
    except Exception:
        return None


def iter_images(root, recursive: bool = True):
    """遍历目录下所有图像文件。"""
    root = Path(root)
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if p.is_file() and is_image(p):
            yield p
