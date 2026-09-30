"""COCO / VOC / LabelMe 适配器测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_adapters.py

覆盖：
    - 三种格式的解析正确性（含坐标换算）
    - 划分识别（文件名 / ImageSets / 目录）
    - 形状处理（LabelMe 的 rectangle / polygon / circle，line 被跳过）
    - VOC 1-based 坐标开关
    - 格式自动探测不串台
    - 端到端往返：三种格式导出为 YOLO 后再接入，数量一致
"""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image  # noqa: E402

from core.export import TASK_DETECTION, ExportConfig, export_yolo  # noqa: E402
from core.ingest import detect_format, load_dataset  # noqa: E402
from core.ir import KIND_BBOX, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL  # noqa: E402
from core.split import SplitConfig, assign_splits  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [ok]   {message}")
    else:
        print(f"  [FAIL] {message}")
        _failures.append(message)


def eq(actual, expected, message: str) -> None:
    check(actual == expected, f"{message} (期望 {expected!r}, 实际 {actual!r})")


def _img(path: Path, size=(64, 48), color=(30, 60, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


# ---------------------------------------------------------------------------
# COCO 夹具
# ---------------------------------------------------------------------------


def make_coco(root: Path) -> None:
    """标准 COCO 布局：annotations/ 放标注，train2017/ val2017/ 放图。"""
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split, n in (("train", 3), ("val", 2)):
        for i in range(n):
            _img(root / f"{split}2017" / f"{split}_{i:03d}.jpg", (64, 48), (i * 30 % 255, 60, 90))

    def build(split, image_ids, anns):
        return {
            "info": {"description": "fixture"},
            "images": [
                {
                    "id": iid,
                    "file_name": f"{split}2017/{split}_{idx:03d}.jpg",
                    "width": 64,
                    "height": 48,
                }
                for iid, idx in image_ids
            ],
            "annotations": anns,
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
        }

    train = build(
        "train",
        [(1, 0), (2, 1), (3, 2)],
        [
            # bbox 是 [x, y, w, h]（左上角原点、绝对像素）
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10], "area": 200, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [0, 0, 8, 8], "area": 64, "iscrowd": 0},
            {"id": 12, "image_id": 2, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0},
        ],
    )
    val = build(
        "val",
        [(4, 0), (5, 1)],
        [{"id": 13, "image_id": 4, "category_id": 2, "bbox": [5, 5, 5, 5], "area": 25, "iscrowd": 1}],
    )

    (ann_dir / "instances_train2017.json").write_text(json.dumps(train), encoding="utf-8")
    (ann_dir / "instances_val2017.json").write_text(json.dumps(val), encoding="utf-8")


# ---------------------------------------------------------------------------
# VOC 夹具
# ---------------------------------------------------------------------------


def _voc_xml(filename: str, size, objects) -> str:
    node = ET.Element("annotation")
    ET.SubElement(node, "folder").text = "VOC2007"
    ET.SubElement(node, "filename").text = filename
    size_node = ET.SubElement(node, "size")
    ET.SubElement(size_node, "width").text = str(size[0])
    ET.SubElement(size_node, "height").text = str(size[1])
    ET.SubElement(size_node, "depth").text = "3"
    for name, (x1, y1, x2, y2), difficult in objects:
        obj = ET.SubElement(node, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "1" if difficult else "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(x1)
        ET.SubElement(box, "ymin").text = str(y1)
        ET.SubElement(box, "xmax").text = str(x2)
        ET.SubElement(box, "ymax").text = str(y2)
    return ET.tostring(node, encoding="unicode")


def make_voc(root: Path) -> None:
    ann = root / "Annotations"
    imgs = root / "JPEGImages"
    sets = root / "ImageSets" / "Main"
    ann.mkdir(parents=True, exist_ok=True)
    imgs.mkdir(parents=True, exist_ok=True)
    sets.mkdir(parents=True, exist_ok=True)

    spec = {
        "000001": ("cat", (10, 5, 29, 14)),
        "000002": ("dog", (0, 0, 20, 20)),
        "000003": ("cat", (5, 5, 15, 15)),
        "000004": ("cat", (1, 1, 9, 9)),
    }
    for stem, (name, box) in spec.items():
        _img(imgs / f"{stem}.jpg")
        (ann / f"{stem}.xml").write_text(
            _voc_xml(f"{stem}.jpg", (64, 48), [(name, box, stem == "000002")]),
            encoding="utf-8",
        )

    (sets / "train.txt").write_text("000001\n000002\n", encoding="utf-8")
    (sets / "val.txt").write_text("000003\n", encoding="utf-8")
    (sets / "test.txt").write_text("000004\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# LabelMe 夹具
# ---------------------------------------------------------------------------


def make_labelme(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _img(root / "img_1.jpg")
    _img(root / "img_2.jpg")
    _img(root / "img_3.jpg")

    (root / "img_1.json").write_text(
        json.dumps(
            {
                "version": "5.1.1",
                "imagePath": "img_1.jpg",
                "imageWidth": 64,
                "imageHeight": 48,
                "shapes": [
                    # 角点故意反序，适配器应自动纠正
                    {"label": "cat", "shape_type": "rectangle", "points": [[30, 15], [10, 5]], "flags": {}},
                    {"label": "dog", "shape_type": "rectangle", "points": [[0, 0], [8, 8]], "flags": {}},
                ],
            }
        ),
        encoding="utf-8",
    )

    (root / "img_2.json").write_text(
        json.dumps(
            {
                "version": "5.1.1",
                "imagePath": "img_2.jpg",
                "imageWidth": 64,
                "imageHeight": 48,
                "shapes": [
                    {"label": "cat", "shape_type": "polygon",
                     "points": [[0, 0], [10, 0], [10, 10], [0, 10]], "flags": {}},
                    {"label": "dog", "shape_type": "circle",
                     "points": [[20, 20], [30, 20]], "flags": {}},
                ],
            }
        ),
        encoding="utf-8",
    )

    (root / "img_3.json").write_text(
        json.dumps(
            {
                "version": "5.1.1",
                "imagePath": "img_3.jpg",
                "imageWidth": 64,
                "imageHeight": 48,
                "shapes": [
                    # line / point 无法构成检测框，应被跳过
                    {"label": "cat", "shape_type": "line", "points": [[0, 0], [10, 10]], "flags": {}},
                    {"label": "cat", "shape_type": "point", "points": [[5, 5]], "flags": {}},
                ],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def test_coco() -> None:
    print("\n== COCO ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "coco_ds"
        make_coco(root)

        eq(detect_format(root), "coco", "自动探测为 coco")
        bundle = load_dataset(root)
        stats = bundle.stats()

        eq(stats["annotation_kind"], KIND_BBOX, "标注形态为检测框")
        eq(stats["num_images"], 5, "图像总数 (3 train + 2 val)")
        eq(stats["num_annotations"], 4, "标注总数")
        eq(stats["split_counts"], {SPLIT_TRAIN: 3, SPLIT_VAL: 2, SPLIT_TEST: 0}, "按文件名识别划分")
        eq(set(stats["count_by_category"]), {"cat", "dog"}, "类别集合")

        # bbox [x,y,w,h] -> xyxy
        target = next(
            im for im in bundle.images.values() if im.path.endswith("train_000.jpg")
        )
        boxes = sorted(a.bbox.as_tuple() for a in bundle.annotations_of(target.uid))
        eq(boxes, [(0.0, 0.0, 8.0, 8.0), (10.0, 5.0, 30.0, 15.0)], "COCO [x,y,w,h] 正确换算为 xyxy")

        # 分组来自 file_name 的目录层级，防止同场景跨子集
        eq(target.group, "train2017", "分组取自路径目录")
        check(all(im.split for im in bundle.images.values()), "所有图像都有划分")

        # iscrowd 被记录
        crowd = [a for a in bundle.annotations if a.group == "iscrowd"]
        eq(len(crowd), 1, "iscrowd 标注被标记")

        # 缺少 width/height 时回退到探测图像尺寸
        check(all(im.width == 64 and im.height == 48 for im in bundle.images.values()),
              "图像尺寸正确")


def test_coco_single_file() -> None:
    print("\n== COCO 单文件布局 ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "coco_single"
        (root / "images").mkdir(parents=True)
        _img(root / "images" / "a.jpg")
        _img(root / "images" / "b.jpg")
        data = {
            # 故意不带 width/height，验证回退探测
            "images": [
                {"id": 1, "file_name": "a.jpg"},
                {"id": 2, "file_name": "b.jpg"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]},
            ],
            "categories": [{"id": 7, "name": "thing"}],
        }
        (root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")

        eq(detect_format(root), "coco", "探测为 coco")
        bundle = load_dataset(root)
        eq(len(bundle.images), 2, "2 张图")
        eq(len(bundle.annotations), 1, "1 个标注")
        check(all(im.width == 64 for im in bundle.images.values()), "尺寸回退探测生效")
        check(all(im.split is None for im in bundle.images.values()), "无法判定划分时留空")


# ---------------------------------------------------------------------------
# VOC
# ---------------------------------------------------------------------------


def test_voc() -> None:
    print("\n== Pascal VOC ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "voc_ds"
        make_voc(root)

        eq(detect_format(root), "voc", "自动探测为 voc")
        bundle = load_dataset(root)
        stats = bundle.stats()

        eq(stats["num_images"], 4, "图像总数")
        eq(stats["num_annotations"], 4, "标注总数")
        eq(
            stats["split_counts"],
            {SPLIT_TRAIN: 2, SPLIT_VAL: 1, SPLIT_TEST: 1},
            "按 ImageSets/Main 识别划分",
        )

        # 原始数值（LabelImg 风格）
        target = next(im for im in bundle.images.values() if im.path.endswith("000001.jpg"))
        box = bundle.annotations_of(target.uid)[0].bbox.as_tuple()
        eq(box, (10.0, 5.0, 29.0, 14.0), "默认直接采用 XML 坐标")

        # difficult 标记被记录
        hard = next(im for im in bundle.images.values() if im.path.endswith("000002.jpg"))
        check(bundle.annotations_of(hard.uid)[0].difficult, "difficult 标记被识别")

        # 1-based 开关
        bundle2 = load_dataset(root, voc_one_based=True)
        t2 = next(im for im in bundle2.images.values() if im.path.endswith("000001.jpg"))
        box2 = bundle2.annotations_of(t2.uid)[0].bbox.as_tuple()
        eq(box2, (9.0, 4.0, 29.0, 14.0), "voc_one_based=True 时做 -1 修正")


def test_voc_without_imagesets() -> None:
    print("\n== VOC 无 ImageSets ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "voc_no_sets"
        make_voc(root)
        for f in (root / "ImageSets" / "Main").glob("*.txt"):
            f.unlink()

        bundle = load_dataset(root)
        eq(len(bundle.images), 4, "仍能读全部图像")
        check(all(im.split is None for im in bundle.images.values()),
              "无 ImageSets 时划分留空，交由划分模块处理")


# ---------------------------------------------------------------------------
# LabelMe
# ---------------------------------------------------------------------------


def test_labelme() -> None:
    print("\n== LabelMe ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "labelme_ds"
        make_labelme(root)

        eq(detect_format(root), "labelme", "自动探测为 labelme")
        bundle = load_dataset(root)
        stats = bundle.stats()

        eq(stats["num_images"], 3, "图像总数")
        eq(stats["num_annotations"], 4, "有效形状数（line/point 被跳过）")

        img1 = next(im for im in bundle.images.values() if im.path.endswith("img_1.jpg"))
        boxes = sorted(a.bbox.as_tuple() for a in bundle.annotations_of(img1.uid))
        eq(boxes, [(0.0, 0.0, 8.0, 8.0), (10.0, 5.0, 30.0, 15.0)],
           "rectangle 两角点反序时自动纠正")

        img2 = next(im for im in bundle.images.values() if im.path.endswith("img_2.jpg"))
        anns = {a.category + "|" + a.meta.get("shape_type", ""): a
                for a in bundle.annotations_of(img2.uid)}
        polygon = next(a for a in anns.values() if a.meta["shape_type"] == "polygon")
        eq(polygon.bbox.as_tuple(), (0.0, 0.0, 10.0, 10.0), "polygon 外接矩形")
        check(polygon.segmentation is not None, "polygon 保留 segmentation")

        circle = next(a for a in anns.values() if a.meta["shape_type"] == "circle")
        eq(circle.bbox.as_tuple(), (10.0, 10.0, 30.0, 30.0), "circle 由圆心半径换算外接矩形")

        # line / point 被跳过并告警
        check(any("无法构成检测框" in w for w in bundle.warnings), "对跳过的形状给出警告")
        img3 = next(im for im in bundle.images.values() if im.path.endswith("img_3.jpg"))
        eq(len(bundle.annotations_of(img3.uid)), 0, "只有 line/point 的图无有效标注")


# ---------------------------------------------------------------------------
# 交叉探测
# ---------------------------------------------------------------------------


def test_detection_does_not_cross() -> None:
    print("\n== 格式探测不串台 ==")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        coco_root = base / "coco_ds"
        voc_root = base / "voc_ds"
        lm_root = base / "labelme_ds"
        make_coco(coco_root)
        make_voc(voc_root)
        make_labelme(lm_root)

        eq(detect_format(coco_root), "coco", "COCO 不会被判为 LabelMe（其中也有 json）")
        eq(detect_format(voc_root), "voc", "VOC 判定正确")
        eq(detect_format(lm_root), "labelme", "LabelMe 判定正确")

        # 各适配器的探测函数互不误判
        from core.ingest import CocoAdapter, LabelMeAdapter, VocAdapter

        check(not LabelMeAdapter.detect(coco_root), "LabelMe 不认 COCO 的 json")
        check(not CocoAdapter.detect(lm_root), "COCO 不认 LabelMe 的 json")
        check(not VocAdapter.detect(coco_root), "VOC 不认 COCO 目录")


# ---------------------------------------------------------------------------
# 端到端往返
# ---------------------------------------------------------------------------


def test_roundtrip_all_formats() -> None:
    print("\n== 端到端往返（三种格式 -> YOLO -> 再接入）==")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        cases = [
            ("coco", make_coco),
            ("voc", make_voc),
            ("labelme", make_labelme),
        ]

        for name, maker in cases:
            src = base / f"{name}_src"
            maker(src)
            bundle = load_dataset(src)
            images_before = len(bundle.images)
            anns_before = len(bundle.annotations)

            assign_splits(
                bundle,
                SplitConfig(ratios=(0.6, 0.2, 0.2), seed=42,
                            respect_existing=False, min_val=0, min_test=0),
            )

            out = base / f"{name}_out"
            report = export_yolo(bundle, out, ExportConfig(task=TASK_DETECTION))
            check(report.task == TASK_DETECTION, f"{name}: 导出为检测任务")

            reloaded = load_dataset(out)
            eq(reloaded.stats()["num_images"], images_before, f"{name}: 往返图像数一致")
            eq(reloaded.stats()["num_annotations"], anns_before, f"{name}: 往返标注数一致")
            eq(reloaded.stats()["annotation_kind"], KIND_BBOX, f"{name}: 往返形态一致")
            eq(
                sorted(reloaded.stats()["count_by_category"].items()),
                sorted(bundle.stats()["count_by_category"].items()),
                f"{name}: 往返类别分布一致",
            )


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 适配器测试（COCO / VOC / LabelMe）")
    test_coco()
    test_coco_single_file()
    test_voc()
    test_voc_without_imagesets()
    test_labelme()
    test_detection_does_not_cross()
    test_roundtrip_all_formats()

    print("\n" + "=" * 60)
    if _failures:
        print(f"失败 {len(_failures)} 项:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
