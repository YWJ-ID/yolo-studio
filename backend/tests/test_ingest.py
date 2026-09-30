"""接入层测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_ingest.py

覆盖：
    - YOLO 适配器（临时构造的数据集，自包含）
    - OpenLABEL / VCD 适配器（DMD 夹具，由 make_dmd_fixture.py 生成）
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.ingest import (  # noqa: E402
    OpenLabelAdapter,
    available_adapters,
    detect_format,
    extract_frame_index,
    load_dataset,
)
from core.ir import KIND_BBOX, KIND_IMAGE  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dmd_sample"

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [ok]   {message}")
    else:
        print(f"  [FAIL] {message}")
        _failures.append(message)


def eq(actual, expected, message: str) -> None:
    check(actual == expected, f"{message} (期望 {expected!r}, 实际 {actual!r})")


# ---------------------------------------------------------------------------


def test_registry() -> None:
    print("\n== 适配器注册 ==")
    names = [a["name"] for a in available_adapters()]
    check("yolo" in names, f"已注册 yolo 适配器: {names}")
    check("openlabel" in names, f"已注册 openlabel 适配器: {names}")


def test_frame_index() -> None:
    print("\n== 帧号提取 ==")
    eq(extract_frame_index("000059"), 59, "纯数字文件名")
    eq(extract_frame_index("sessionA_frame_000123"), 123, "带前缀的文件名")
    eq(extract_frame_index("frame-7"), 7, "带分隔符")
    eq(extract_frame_index("no_digits"), None, "无数字返回 None")


def test_openlabel_format_detection() -> None:
    print("\n== OpenLABEL 格式探测 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, f"夹具不存在: {FIXTURE_DIR}，请先运行 make_dmd_fixture.py")
        return
    eq(detect_format(FIXTURE_DIR), "openlabel", "自动探测到 openlabel 格式")
    check(OpenLabelAdapter.detect(FIXTURE_DIR), "OpenLabelAdapter.detect 为真")


def test_openlabel_load() -> None:
    print("\n== OpenLABEL 读取（driver_actions 层级）==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return

    bundle = load_dataset(FIXTURE_DIR, group_by=None)
    stats = bundle.stats()

    eq(stats["format"], "openlabel", "format")
    eq(stats["annotation_kind"], KIND_IMAGE, "标注形态为图像级")
    eq(stats["num_images"], 60, "帧图像数量")
    eq(stats["num_bbox_annotations"], 0, "不含边界框标注")
    eq(stats["num_image_labels"], 60, "图像级标注数量")

    counts = stats["count_by_category"]
    eq(counts.get("safe_drive"), 35, "safe_drive 帧数 (0-9 + 35-59)")
    eq(counts.get("texting_right"), 20, "texting_right 帧数 (10-29)")
    eq(counts.get("drinking"), 5, "drinking 帧数 (30-34)")

    check(stats["warnings"] == 0, f"无接入警告 (实际 {stats['warnings']})")

    # 分组 = 视频名，用于防泄漏划分
    groups = {im.group for im in bundle.images.values()}
    eq(len(groups), 1, "全部帧属于同一个 group")
    check(all(g for g in groups), f"group 非空: {groups}")

    # 逐帧核对边界，确认区间展开正确
    by_frame = {}
    for im in bundle.images.values():
        labels = [a.category for a in bundle.annotations_of(im.uid)]
        by_frame[im.meta["frame_index"]] = labels

    eq(by_frame.get(0), ["safe_drive"], "帧 0 归属")
    eq(by_frame.get(9), ["safe_drive"], "帧 9 归属（区间右端点）")
    eq(by_frame.get(10), ["texting_right"], "帧 10 归属（区间左端点）")
    eq(by_frame.get(29), ["texting_right"], "帧 29 归属")
    eq(by_frame.get(30), ["drinking"], "帧 30 归属")
    eq(by_frame.get(34), ["drinking"], "帧 34 归属")
    eq(by_frame.get(35), ["safe_drive"], "帧 35 归属")
    eq(by_frame.get(59), ["safe_drive"], "帧 59 归属（末帧）")

    # 标注形态
    sample = bundle.annotations[0]
    eq(sample.kind, KIND_IMAGE, "标注 kind")
    check(sample.bbox is None, "图像级标注的 bbox 为 None")
    check(not sample.is_bbox, "is_bbox 为假")


def test_openlabel_level_selection() -> None:
    print("\n== OpenLABEL 层级选择 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return

    # 换层级：gaze_on_road
    gaze = load_dataset(FIXTURE_DIR, level="gaze_on_road")
    eq(gaze.stats()["num_images"], 60, "gape 层级图像数")
    eq(list(gaze.stats()["count_by_category"].keys()), ["looking_road"], "只取到该层级的标签")

    # level=None：保留完整 semantic_type
    all_levels = load_dataset(FIXTURE_DIR, level=None)
    cats = set(all_levels.stats()["count_by_category"].keys())
    check("driver_actions/safe_drive" in cats, f"level=None 保留完整 type: {sorted(cats)}")
    check("gaze_on_road/looking_road" in cats, "level=None 包含 gaze 层级")

    # include_objects：并入 object 类型标注
    with_obj = load_dataset(FIXTURE_DIR, include_objects=True)
    cats_obj = with_obj.stats()["count_by_category"]
    eq(cats_obj.get("cellphone"), 20, "objects_in_scene 的 cellphone 帧数")
    eq(with_obj.stats()["num_annotations"], 80, "总标注数 = 60 动作 + 20 物体")


def test_yolo_adapter() -> None:
    print("\n== YOLO 适配器（临时数据集）==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "mini_ds"
        for split, n in (("train", 3), ("valid", 2), ("test", 1)):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
            for i in range(n):
                name = f"{split}_{i}"
                # 1x1 最小合法 JPEG（PIL 生成）
                from PIL import Image

                Image.new("RGB", (32, 24), (i * 20 % 256, 0, 0)).save(
                    root / split / "images" / f"{name}.jpg"
                )
                (root / split / "labels" / f"{name}.txt").write_text(
                    "0 0.5 0.5 0.25 0.25\n1 0.25 0.25 0.1 0.1\n", encoding="utf-8"
                )
        (root / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n"
            "nc: 2\nnames:\n  0: cat\n  1: dog\n",
            encoding="utf-8",
        )

        eq(detect_format(root), "yolo", "自动探测到 yolo 格式")
        bundle = load_dataset(root)
        stats = bundle.stats()

        eq(stats["num_images"], 6, "图像总数")
        eq(stats["num_annotations"], 12, "标注总数 (每图 2 框)")
        eq(stats["annotation_kind"], KIND_BBOX, "标注形态为检测框")
        eq(stats["split_counts"], {"train": 3, "val": 2, "test": 1}, "划分统计")
        eq(set(stats["count_by_category"]), {"cat", "dog"}, "类别集合")

        # 坐标换算：xc=0.5,yc=0.5,w=0.25,h=0.25, 图 32x24 => xyxy=(12,9,20,15)
        im = next(im for im in bundle.images.values() if im.width == 32)
        ann = bundle.annotations_of(im.uid)[0]
        eq(ann.kind, KIND_BBOX, "标注 kind 为 bbox")
        eq(tuple(round(v, 3) for v in ann.bbox.as_tuple()), (12.0, 9.0, 20.0, 15.0), "归一化坐标换算")

        # 注册表探测不应把 yolo 误判为 openlabel
        check(not OpenLabelAdapter.detect(root), "YOLO 数据集不会被误判为 OpenLABEL")


def test_browse_api() -> None:
    print("\n== 数据浏览器接口（筛选 + 分页）==")
    from app.api.routes.datasets import browse
    from app.schemas import BrowseOptions, BrowseRequest

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "browse_ds"
        plan = {"train": [0, 0, 0], "valid": [1, 1], "test": [0]}
        for split, classes in plan.items():
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
            for i, cls in enumerate(classes):
                from PIL import Image

                name = f"{split}_{i}"
                Image.new("RGB", (32, 24), (i * 20 % 256, 0, 0)).save(
                    root / split / "images" / f"{name}.jpg"
                )
                (root / split / "labels" / f"{name}.txt").write_text(
                    f"{cls} 0.5 0.5 0.25 0.25\n", encoding="utf-8"
                )
        (root / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n"
            "nc: 2\nnames:\n  0: cat\n  1: dog\n",
            encoding="utf-8",
        )

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(limit=2)))
        eq(res.total, 6, "总数")
        eq(len(res.images), 2, "首页返回 limit 张")
        eq(res.images[0]["source_id"] != "", True, "样本带 source_id")
        eq([s["source_id"] for s in res.sources], [res.images[0]["source_id"]], "来源清单")

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(offset=4, limit=10)))
        eq(len(res.images), 2, "末页数量")
        eq(res.total, 6, "筛选总数不随分页变化")

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(split="train")))
        eq(res.total, 3, "按划分筛选")

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(category="cat")))
        eq(res.total, 4, "按类别筛选")
        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(category="dog")))
        eq(res.total, 2, "按类别筛选（dog）")

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(kind="image")))
        eq(res.total, 0, "按标注形态筛选（无图像级标注）")

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(min_objects=2)))
        eq(res.total, 0, "按目标数下限筛选")

        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(search="train_")))
        eq(res.total, 3, "按路径子串筛选")

        # 夹具图像均为 32x24，长边 32
        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(min_edge=100)))
        eq(res.total, 0, "按图像长边下限筛选（超出）")
        res = browse(BrowseRequest(path=str(root), browse=BrowseOptions(max_edge=40)))
        eq(res.total, 6, "按图像长边上限筛选（命中）")


def test_ir_roundtrip() -> None:
    print("\n== IR 序列化往返 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return

    bundle = load_dataset(FIXTURE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ir.json"
        bundle.save_json(out)
        raw = json.loads(out.read_text(encoding="utf-8"))
        eq(raw["annotations"][0]["bbox"], None, "图像级标注 bbox 序列化为 null")

        from core.ir import DatasetBundle

        restored = DatasetBundle.load_json(out)
        eq(len(restored.images), len(bundle.images), "图像数往返一致")
        eq(len(restored.annotations), len(bundle.annotations), "标注数往返一致")
        eq(restored.stats()["count_by_category"], bundle.stats()["count_by_category"], "类别分布往返一致")


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 接入层测试")
    test_registry()
    test_frame_index()
    test_openlabel_format_detection()
    test_openlabel_load()
    test_openlabel_level_selection()
    test_yolo_adapter()
    test_browse_api()
    test_ir_roundtrip()

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
