"""core.io 包：图像、哈希、路径工具的统一出口。"""

from __future__ import annotations

from .hash import (
    find_near_duplicates,
    hamming_distance,
    phash_bits,
    sha1_file,
)
from .image import (
    IMAGE_EXTS,
    get_exif_orientation,
    is_image,
    is_readable,
    iter_images,
    probe_size,
)

__all__ = [
    "IMAGE_EXTS",
    "find_near_duplicates",
    "get_exif_orientation",
    "hamming_distance",
    "is_image",
    "is_readable",
    "iter_images",
    "phash_bits",
    "probe_size",
    "sha1_file",
]
