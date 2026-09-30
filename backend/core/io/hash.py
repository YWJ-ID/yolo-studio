"""哈希工具：精确去重（sha1）与近似去重（感知哈希）。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image

_CHUNK = 1 << 20  # 1MB


def sha1_file(path, chunk_size: int = _CHUNK) -> Optional[str]:
    """文件内容 sha1，用于精确重复检测。"""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _to_bits(value: int, size: int) -> str:
    return bin(value)[2:].zfill(size)[-size:]


def phash_bits(path, hash_size: int = 8, highfreq_factor: int = 4) -> Optional[str]:
    """感知哈希（DCT 版），返回十六进制字符串。

    用于发现"同一张图被裁剪/压缩/轻微改动后重复"的情况——
    这是数据集里最隐蔽的泄漏来源。
    """
    try:
        img_size = hash_size * highfreq_factor
        with Image.open(path) as im:
            im = im.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
            import numpy as np

            pixels = np.asarray(im, dtype=float)

        # 二维 DCT（用矩阵乘法实现，避免依赖 scipy）
        dct_mat = _dct_matrix(img_size)
        dct = dct_mat @ pixels @ dct_mat.T
        low = dct[:hash_size, :hash_size]
        flat = low.flatten()
        med = float(np.median(flat[1:]))  # 去掉直流分量
        bits = "".join("1" if v > med else "0" for v in flat)
        return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
    except Exception:
        return None


def _dct_matrix(n: int):
    """生成 n x n 的 DCT-II 变换矩阵。"""
    import math

    import numpy as np

    mat = np.zeros((n, n), dtype=float)
    factor = math.pi / (2 * n)
    for k in range(n):
        alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return mat


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """两个 phash 的汉明距离，越小越相似。"""
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return 1 << 30
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def find_near_duplicates(
    items: Iterable[tuple],
    max_distance: int = 6,
) -> List[tuple]:
    """在 (uid, phash) 列表中找近似重复对。

    朴素做法是两两比较，复杂度 O(n²)——几万张图就是几千万次比较，实际不可用。
    这里用**多重索引**加速：把 64 位哈希切成 k 段并建立倒排索引。
    由鸽巢原理，若两个哈希的汉明距离 < k，则至少有一段完全相同，
    因此只需比较「至少共享一段」的候选对，把候选集缩到极小。

    k 取 max_distance + 1，保证不漏。
    """
    entries = [(uid, ph) for uid, ph in items if ph]
    if not entries:
        return []
    if len(entries) <= 1:
        return []

    # 哈希位数由字符串长度推出（hex，每字符 4 位）
    hash_len = len(entries[0][1])
    bits = hash_len * 4

    k = max(2, max_distance + 1)
    segments = _segment_sizes(bits, k)

    buckets: Dict[tuple, List[int]] = {}
    for idx, (_uid, ph) in enumerate(entries):
        value = int(ph, 16)
        offset = 0
        for seg_idx, size in enumerate(segments):
            chunk = (value >> (bits - offset - size)) & ((1 << size) - 1)
            buckets.setdefault((seg_idx, chunk), []).append(idx)
            offset += size

    candidates: set = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((members[i], members[j]))

    dupes: List[tuple] = []
    for i, j in candidates:
        if _hamming_hex(entries[i][1], entries[j][1]) <= max_distance:
            dupes.append((entries[i][0], entries[j][0]))
    return dupes


def _segment_sizes(bits: int, k: int) -> List[int]:
    """把 bits 位尽量均匀地切成 k 段。"""
    base = bits // k
    extra = bits % k
    return [base + (1 if i < extra else 0) for i in range(k)]


def _hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 1 << 30
    return bin(int(a, 16) ^ int(b, 16)).count("1")
