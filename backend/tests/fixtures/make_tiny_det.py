"""生成一个极小的目标检测数据集，用于真实训练冒烟验证。

真训练（即使是 CPU）在 CI/测试里太慢，因此测试用假训练进程；
但"命令能真的拉起 ultralytics 并写出 results.csv"必须真跑一次，
这个数据集就是给那次真跑准备的：8 张 train + 4 张 val，2 个类别。

用法：
    cd backend
    .\.venv\Scripts\python.exe tests\fixtures\make_tiny_det.py
    .\.venv\Scripts\python.exe -m cli.main train --data storage\demo_sources\tiny_det\data.yaml \
        --weights yolo11n.yaml --epochs 1 --imgsz 32 --batch 4 --device cpu --name smoke_yolo11n

图像用"位置与大小各不相同的小方块/圆"制造差异，
不靠颜色区分——JPEG 量化会让相邻色号编码出相同字节（见 PROGRESS R-16）。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parents[2] / "storage" / "demo_sources" / "tiny_det"
IMAGE_SIZE = 64
CLASSES = ("square", "circle")

# (子集, 文件名, 类别, 归一化 bbox: xc yc w h)
PLAN = [
    ("train", "t0", 0, (0.25, 0.25, 0.30, 0.30)),
    ("train", "t1", 1, (0.70, 0.30, 0.25, 0.25)),
    ("train", "t2", 0, (0.50, 0.75, 0.40, 0.20)),
    ("train", "t3", 1, (0.15, 0.60, 0.20, 0.30)),
    ("train", "t4", 0, (0.80, 0.80, 0.20, 0.20)),
    ("train", "t5", 1, (0.40, 0.50, 0.35, 0.35)),
    ("train", "t6", 0, (0.30, 0.45, 0.15, 0.15)),
    ("train", "t7", 1, (0.60, 0.10, 0.30, 0.15)),
    ("val", "v0", 0, (0.30, 0.30, 0.25, 0.25)),
    ("val", "v1", 1, (0.65, 0.65, 0.30, 0.30)),
    ("val", "v2", 0, (0.50, 0.20, 0.20, 0.20)),
    ("val", "v3", 1, (0.20, 0.75, 0.25, 0.20)),
    ("test", "s0", 0, (0.35, 0.35, 0.25, 0.25)),
    ("test", "s1", 1, (0.60, 0.60, 0.25, 0.25)),
]


def _to_pixels(box):
    xc, yc, w, h = box
    x1 = (xc - w / 2) * IMAGE_SIZE
    y1 = (yc - h / 2) * IMAGE_SIZE
    x2 = (xc + w / 2) * IMAGE_SIZE
    y2 = (yc + h / 2) * IMAGE_SIZE
    return x1, y1, x2, y2


def _draw(path: Path, cls: int, box) -> None:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = _to_pixels(box)
    if cls == 0:
        draw.rectangle([x1, y1, x2, y2], fill=(230, 200, 60))
    else:
        draw.ellipse([x1, y1, x2, y2], fill=(90, 180, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)


def main() -> None:
    labels_root = OUT_DIR / "labels"
    images_root = OUT_DIR / "images"
    for split in ("train", "val", "test"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    for split, stem, cls, box in PLAN:
        _draw(images_root / split / f"{stem}.jpg", cls, box)
        (labels_root / split / f"{stem}.txt").write_text(
            f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n",
            encoding="utf-8",
        )

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    # path 必须写绝对路径：ultralytics 对相对 path 的解析基准不是 yaml 所在目录
    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )

    print(f"极小检测数据集已生成: {OUT_DIR}")
    for split in ("train", "val", "test"):
        print(f"  {split:<5}: {sum(1 for p in PLAN if p[0] == split)} 张")
    print(f"  类别 : {', '.join(CLASSES)}")


if __name__ == "__main__":
    main()
