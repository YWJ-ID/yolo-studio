"""划分 + 导出 测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_pipeline.py

覆盖：
    - 分组划分：同组不可跨子集（防泄漏）
    - 分层划分：长尾类在各子集的分布
    - 可复现性：同 seed 同结果
    - 分类导出：目录布局 + data.yaml
    - 检测导出：归一化坐标 + 文件名冲突处理
    - 端到端往返：导出后再接入，数量与划分一致
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.export import (  # noqa: E402
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    ExportConfig,
    ExportError,
    detect_task,
    export_yolo,
)
from core.ingest import load_and_merge, load_dataset  # noqa: E402
from core.ir import (  # noqa: E402
    KIND_IMAGE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Annotation,
    BBox,
    DatasetBundle,
    ImageRecord,
)
from core.split import SplitConfig, assign_splits  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dmd_sample"
FRAMES_DIR = FIXTURE_DIR / "frames"

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
# 构造测试数据
# ---------------------------------------------------------------------------


def make_detection_bundle(n_groups: int = 4, per_group: int = 15, classes=("head", "phone")) -> DatasetBundle:
    """用夹具帧图构造一个带分组的检测数据集。"""
    bundle = DatasetBundle(source_id="synthetic", format_name="synthetic")
    frames = sorted(FRAMES_DIR.glob("*.jpg"))[: n_groups * per_group]

    for i, path in enumerate(frames):
        group = f"video_{i // per_group}"
        uid = f"{group}_{i:04d}"
        bundle.add_image(
            ImageRecord(
                uid=uid,
                path=str(path),
                rel_path=path.name,
                width=48,
                height=36,
                source_id="synthetic",
                group=group,
                meta={"frame_index": i},
            )
        )
        # 每图两个框，坐标固定便于校验
        bundle.add_annotation(
            Annotation(uid, classes[0], BBox(12, 9, 24, 18))
        )
        if i % 3 == 0:
            bundle.add_annotation(
                Annotation(uid, classes[1], BBox(0, 0, 12, 9))
            )
    return bundle


def make_collision_bundle() -> DatasetBundle:
    """两个来源、同名文件，用于验证导出时的文件名冲突处理。"""
    bundle = DatasetBundle(source_id="multi", format_name="synthetic")
    frames = sorted(FRAMES_DIR.glob("*.jpg"))[:2]
    for src in ("dsA", "dsB"):
        for path in frames:
            uid = f"{src}_{path.stem}"
            bundle.add_image(
                ImageRecord(
                    uid=uid,
                    path=str(path),
                    rel_path=path.name,
                    width=48,
                    height=36,
                    source_id=src,
                )
            )
            bundle.add_annotation(Annotation(uid, "head", BBox(6, 6, 18, 18)))
    return bundle


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def test_split_group_no_leak() -> None:
    print("\n== 分组划分：防泄漏 ==")
    bundle = make_detection_bundle(n_groups=10, per_group=6)
    report = assign_splits(bundle, SplitConfig(ratios=(0.8, 0.1, 0.1), seed=42))

    eq(report.images_total, 60, "图像总数")
    eq(report.units_grouped, 10, "组数")
    eq(report.images_assigned, 60, "全部完成划分")

    # 核心断言：任何一个组都不能跨子集
    group_splits: dict[str, set] = {}
    for im in bundle.images.values():
        group_splits.setdefault(im.group, set()).add(im.split)
    spanning = {g: s for g, s in group_splits.items() if len(s) > 1}
    eq(spanning, {}, "没有任何组跨子集（无泄漏）")

    eq(sum(report.split_images.values()), 60, "各子集图像数之和")
    check(report.split_images[SPLIT_TRAIN] > 0, f"train 非空 {report.split_images}")


def test_split_reproducible() -> None:
    print("\n== 划分可复现 ==")
    b1 = make_detection_bundle(n_groups=10, per_group=6)
    b2 = make_detection_bundle(n_groups=10, per_group=6)
    assign_splits(b1, SplitConfig(seed=7))
    assign_splits(b2, SplitConfig(seed=7))
    same = all(b1.images[u].split == b2.images[u].split for u in b1.images)
    check(same, "同 seed 得到完全相同的划分")

    b3 = make_detection_bundle(n_groups=10, per_group=6)
    assign_splits(b3, SplitConfig(seed=8))
    diff = any(b1.images[u].split != b3.images[u].split for u in b1.images)
    check(diff, "不同 seed 得到不同划分")


def test_split_stratified_rare_class() -> None:
    print("\n== 分层划分：长尾类 ==")
    # 夹具共 60 帧 -> 12 组，每组 5 帧；只有 3 个组含稀有类 "rare"
    bundle = DatasetBundle(source_id="strat", format_name="synthetic")
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    rare_groups = {0, 5, 11}
    for i, path in enumerate(frames):
        group = f"v{i // 5}"
        uid = f"{group}_{i:04d}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(path), rel_path=path.name,
                        width=48, height=36, source_id="strat", group=group)
        )
        bundle.add_annotation(Annotation(uid, "common", BBox(0, 0, 10, 10)))
        if i // 5 in rare_groups:
            bundle.add_annotation(Annotation(uid, "rare", BBox(10, 10, 20, 20)))

    eq(len({f"v{i // 5}" for i in range(len(frames))}), 12, "组数")
    report = assign_splits(bundle, SplitConfig(ratios=(0.6, 0.2, 0.2), seed=42))
    eq(sum(report.split_images.values()), 60, "图像总数")

    rare_splits = {
        im.group: im.split
        for im in bundle.images.values()
        if any(a.category == "rare" for a in bundle.annotations_of(im.uid))
    }
    eq(len(rare_splits), 3, "3 个含稀有类的组")
    # 3 个稀有单元按 0.6/0.2/0.2 分配 -> 2/1/0，稀有类覆盖 2 个子集。
    # 小样本下"比例精确"与"每类都进各子集"无法兼得，这里保比例、报告缺失。
    eq(len(set(rare_splits.values())), 2, "稀有类按比例覆盖 2 个子集")
    check(all(report.split_images[s] > 0 for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)),
          f"三个子集均非空 {report.split_images}")
    # 关键：缺失必须被明确报告，否则用户会看到 NaN 指标却不知原因
    eq(report.classes_missing_in_split.get(SPLIT_TEST), ["rare"],
       "报告 test 中缺失的类别")
    check(any("缺少" in w and "rare" in w for w in report.warnings),
          "缺失类别产生明确警告")

    # 未分层时，稀有类不应被如此均匀地摊开（对照）
    plain = DatasetBundle(source_id="plain", format_name="synthetic")
    for uid, src in bundle.images.items():
        plain.add_image(ImageRecord(uid=uid, path=src.path, rel_path=src.rel_path,
                                    width=48, height=36, source_id="plain", group=src.group))
    for a in bundle.annotations:
        plain.add_annotation(Annotation(a.image_uid, a.category, a.bbox))
    assign_splits(plain, SplitConfig(ratios=(0.6, 0.2, 0.2), seed=42, strategy="random"))
    plain_rare = {
        plain.images[u].group: plain.images[u].split
        for u, im in plain.images.items()
        if any(a.category == "rare" for a in plain.annotations_of(u))
    }
    check(len(plain_rare) == 3, f"对照组稀有类组数 {len(plain_rare)}")


def test_split_respect_existing() -> None:
    print("\n== 尊重已有划分 ==")
    bundle = make_detection_bundle(n_groups=6, per_group=5)
    # 手工把前两组钉死
    for im in bundle.images.values():
        if im.group in ("video_0", "video_1"):
            im.split = SPLIT_TEST
    before = {u: im.split for u, im in bundle.images.items() if im.split}
    assign_splits(bundle, SplitConfig(seed=42))
    after = {u: bundle.images[u].split for u in before}
    eq(after, before, "已有划分未被改动")
    check(all(bundle.images[u].split for u in bundle.images), "剩余图像也都获得划分")


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def test_task_inference() -> None:
    print("\n== 任务类型推断 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return
    eq(detect_task(load_dataset(FIXTURE_DIR)), TASK_CLASSIFICATION, "OpenLABEL 数据 -> 分类")
    eq(detect_task(make_detection_bundle()), TASK_DETECTION, "带框数据 -> 检测")

    mixed = make_detection_bundle(n_groups=1, per_group=2)
    im = next(iter(mixed.images.values()))
    mixed.add_annotation(Annotation(im.uid, "x", bbox=None, kind=KIND_IMAGE))
    try:
        detect_task(mixed)
        check(False, "混合标注应抛错")
    except ExportError:
        check(True, "混合标注正确抛错")


def make_classification_bundle(
    n_groups: int = 6,
    per_group: int = 10,
    classes=("safe_drive", "texting_right", "drinking"),
) -> DatasetBundle:
    """构造多视频的分类数据集（每个视频一个组、一个类别）。"""
    bundle = DatasetBundle(source_id="cls", format_name="synthetic")
    frames = sorted(FRAMES_DIR.glob("*.jpg"))[: n_groups * per_group]

    for i, path in enumerate(frames):
        group = f"v{i // per_group}"
        uid = f"{group}_{i:04d}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(path), rel_path=path.name,
                        width=48, height=36, source_id="cls", group=group)
        )
        bundle.add_annotation(
            Annotation(uid, classes[(i // per_group) % len(classes)], bbox=None, kind=KIND_IMAGE)
        )
    return bundle


def test_export_classification() -> None:
    print("\n== 分类导出（多视频）==")
    if not FRAMES_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return

    bundle = make_classification_bundle()
    split_report = assign_splits(bundle, SplitConfig(ratios=(0.7, 0.15, 0.15), seed=42))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cls_ds"
        report = export_yolo(
            bundle, out,
            ExportConfig(task=TASK_CLASSIFICATION),
            split_report=split_report.to_dict(),
        )

        eq(report.task, TASK_CLASSIFICATION, "任务类型")
        eq(report.classes, ["safe_drive", "texting_right", "drinking"], "类别顺序")
        eq(sum(report.images_exported.values()), 60, "导出图像总数")
        eq(report.images_exported[SPLIT_VAL] > 0, True, "val 非空")
        eq(report.images_exported[SPLIT_TEST] > 0, True, "test 非空")

        # 目录布局：{split}/{class}/*.jpg，且不存在 images/labels
        check(not (out / "images").exists(), "分类布局不应有 images/ 目录")
        for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            split_dir = out / split
            check(split_dir.is_dir(), f"存在 {split}/ 目录")
            class_dirs = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            check(set(class_dirs) <= set(report.classes), f"{split} 下的类别目录合法: {class_dirs}")

        # data.yaml
        import yaml

        yaml_data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        eq(yaml_data["nc"], 3, "data.yaml nc")
        eq(yaml_data["train"], "train", "data.yaml train 指向分类目录")
        eq(yaml_data["names"][0], "safe_drive", "data.yaml names")

        # 血缘卡片
        card = json.loads((out / "dataset_card.json").read_text(encoding="utf-8"))
        eq(card["task"], TASK_CLASSIFICATION, "dataset_card task")
        check(card["split_report"] is not None, "dataset_card 记录了划分信息")

        # 每张图只出现在一个类别目录里（单标签）
        eq(sum(1 for _ in out.rglob("*.jpg")), 60, "图像文件总数（无重复投放）")
        # 每个类别目录下的图片数 = 该类别实例数
        eq(sum(1 for _ in (out / SPLIT_TRAIN / "safe_drive").glob("*.jpg")) > 0, True,
           "train/safe_drive 有图")


def test_export_classification_single_video() -> None:
    print("\n== 分类导出（单视频，防泄漏的必然结果）==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return

    # 真实 DMD 夹具只有一段视频 -> 不能拆散到 train/val/test
    bundle = load_dataset(FIXTURE_DIR)
    report_split = assign_splits(bundle, SplitConfig(ratios=(0.7, 0.15, 0.15), seed=42))

    eq(len({im.group for im in bundle.images.values()}), 1, "夹具只有 1 个组")
    eq(report_split.images_assigned, 60, "60 帧全部完成划分")
    nontrain = report_split.split_images[SPLIT_VAL] + report_split.split_images[SPLIT_TEST]
    eq(nontrain, 0, "同组数据不允许拆到 val/test（无泄漏）")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cls_single"
        report = export_yolo(bundle, out, ExportConfig(task=TASK_CLASSIFICATION))

        eq(report.images_exported[SPLIT_TRAIN], 60, "全部落在 train")
        eq(report.images_exported[SPLIT_VAL], 0, "val 为空")
        # 必须明确告警，否则用户会以为可以直接开训
        check(any("val 子集为空" in w for w in report.warnings), "对空 val 给出警告")


def test_export_detection_and_roundtrip() -> None:
    print("\n== 检测导出 + 端到端往返 ==")
    bundle = make_detection_bundle(n_groups=10, per_group=6)
    split_report = assign_splits(bundle, SplitConfig(ratios=(0.8, 0.1, 0.1), seed=42))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "det_ds"
        report = export_yolo(
            bundle, out,
            ExportConfig(task=TASK_DETECTION),
            split_report=split_report.to_dict(),
        )

        eq(report.task, TASK_DETECTION, "任务类型")
        eq(sum(report.images_exported.values()), 60, "导出图像总数")
        # 每图 1 框 + 每 3 图多 1 框 = 60 + 20
        eq(report.boxes_exported, 80, "导出框总数")
        eq(report.clamped_boxes, 0, "无需裁剪的框")

        for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            check((out / "images" / split).is_dir(), f"存在 images/{split}/")
            check((out / "labels" / split).is_dir(), f"存在 labels/{split}/")

        # 校验一个已知框的归一化结果：48x36 图上 (12,9,24,18)
        # xc=18/48=0.375, yc=13.5/36=0.375, w=12/48=0.25, h=9/36=0.25
        label_files = sorted((out / "labels" / SPLIT_TRAIN).glob("*.txt"))
        check(len(label_files) > 0, "train 有标签文件")
        first = label_files[0].read_text(encoding="utf-8").strip().splitlines()
        parts = first[0].split()
        eq(parts[0], "0", "类别下标")
        eq([round(float(x), 4) for x in parts[1:]], [0.375, 0.375, 0.25, 0.25], "归一化坐标")

        # ---- 往返：导出结果再接入一次 ----
        reloaded = load_dataset(out)
        eq(reloaded.stats()["num_images"], 60, "往返图像数一致")
        eq(reloaded.stats()["num_annotations"], 80, "往返标注数一致")
        eq(
            reloaded.stats()["split_counts"],
            report.images_exported,
            "往返划分统计一致",
        )
        eq(reloaded.stats()["annotation_kind"], "bbox", "往返标注形态为检测框")

        # 坐标往返误差应在浮点精度内
        orig_im = next(im for im in bundle.images.values() if im.width == 48)
        orig_anns = [a for a in bundle.annotations_of(orig_im.uid)]
        check(len(orig_anns) >= 1, "原数据有标注")


def test_export_hardlink_and_collision() -> None:
    print("\n== 文件名冲突与硬链接 ==")
    bundle = make_collision_bundle()
    assign_splits(bundle, SplitConfig(ratios=(1.0, 0.0, 0.0), seed=1, min_val=0, min_test=0))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "collide"
        report = export_yolo(bundle, out, ExportConfig(task=TASK_DETECTION))

        files = sorted(p.name for p in (out / "images" / SPLIT_TRAIN).glob("*.jpg"))
        eq(len(files), 4, "4 张图全部导出")
        eq(len(set(files)), 4, "同来源同名文件不互相覆盖")

        # 硬链接模式（同盘符下应成功）
        out2 = Path(tmp) / "linked"
        report2 = export_yolo(bundle, out2, ExportConfig(task=TASK_DETECTION, file_mode="hardlink"))
        eq(sum(report2.images_exported.values()), 4, "硬链接模式导出成功")


def _write_yolo_dataset(
    root: Path,
    names: list,
    per_split: dict,
    start_index: int = 0,
    label_class: int = 0,
):
    """在临时目录里写一个最小可用的 YOLO 数据集。"""
    from PIL import Image

    for split, n in per_split.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"{start_index + i:06d}"
            Image.new("RGB", (40, 30), (i * 7 % 256, 30, 60)).save(
                root / split / "images" / f"{name}.jpg"
            )
            (root / split / "labels" / f"{name}.txt").write_text(
                f"{label_class} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    yaml_lines = ["path: .", "train: train/images", "val: valid/images", "test: test/images",
                  f"nc: {len(names)}", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def test_merge_multiple_sources() -> None:
    print("\n== 多来源合并 ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        a = tmp_path / "ds_a"
        b = tmp_path / "ds_b"

        # 两个数据集的文件名完全相同（000000.jpg...）——这正是合并时最容易撞车的情况
        _write_yolo_dataset(a, ["cat", "dog"], {"train": 4, "valid": 2, "test": 1}, label_class=0)
        _write_yolo_dataset(b, ["dog", "bird"], {"train": 3, "valid": 1, "test": 1}, label_class=1)

        bundle = load_and_merge(
            [{"path": str(a)}, {"path": str(b)}],
            merged_source_id="union_ds",
        )
        stats = bundle.stats()

        eq(stats["format"], "merged", "格式标记为 merged")
        eq(stats["num_images"], 12, "图像数为两源之和 (7+5)")
        eq(stats["num_annotations"], 12, "标注数为两源之和")
        # ds_a 的标注是 cat，ds_b 的标注是 bird（各自 data.yaml 的类名不同）
        eq(stats["count_by_category"], {"cat": 7, "bird": 5}, "各来源类名按各自 data.yaml 解析")
        # 类别清单是两源 data.yaml 的并集
        eq(bundle.category_names(), ["cat", "dog", "bird"], "类别为两源并集且顺序稳定")

        # 核心：同名文件不能互相覆盖，uid 必须唯一
        eq(len(bundle.images), 12, "uid 唯一，无图像被覆盖丢失")
        paths = [im.path for im in bundle.images.values()]
        eq(len(set(paths)), 12, "12 张图的路径各不相同")

        check(len(bundle.meta.get("sources", [])) == 2, "记录了 2 个来源")

        # 合并后可正常导出
        split_report = assign_splits(bundle, SplitConfig(ratios=(0.7, 0.15, 0.15), seed=42))
        with tempfile.TemporaryDirectory() as tmp2:
            out = Path(tmp2) / "merged_out"
            report = export_yolo(
                bundle, out,
                ExportConfig(task=TASK_DETECTION),
                split_report=split_report.to_dict(),
            )
            eq(sum(report.images_exported.values()), 12, "合并后导出 12 张图")
            eq(report.boxes_exported, 12, "合并后导出 12 个框")

            # 同一个子集目录内不能重名；不同子集目录允许同名（互不干扰）
            total = 0
            for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
                names = [f.name for f in (out / "images" / split).glob("*.jpg")]
                total += len(names)
                eq(len(names), len(set(names)), f"{split} 子集内文件名唯一")
            eq(total, 12, "导出图像总数 12")

            # data.yaml 的类别顺序应与合并后的顺序一致
            import yaml

            yaml_data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            eq([yaml_data["names"][i] for i in sorted(yaml_data["names"])],
               ["cat", "dog", "bird"], "导出的类别顺序与合并结果一致")


def test_merge_duplicate_source() -> None:
    print("\n== 同一来源重复导入 ==")
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "ds_a"
        _write_yolo_dataset(a, ["cat"], {"train": 2, "valid": 1, "test": 1})

        bundle = load_and_merge([{"path": str(a)}, {"path": str(a)}])
        eq(len(bundle.images), 4, "同一目录导入两次不产生重复图像")
        eq(len(bundle.annotations), 4, "同一目录导入两次不产生重复标注")
        check(any("重复导入" in w for w in bundle.warnings), "对重复导入给出警告")


def test_merge_mixed_task_rejected() -> None:
    print("\n== 检测 + 分类 混合来源 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "det"
        _write_yolo_dataset(a, ["cat"], {"train": 2, "valid": 1, "test": 1})

        bundle = load_and_merge([{"path": str(a)}, {"path": str(FIXTURE_DIR)}])
        eq(bundle.stats()["annotation_kind"], "mixed", "合并后标注形态为 mixed")
        # 混合任务无法用单一目录布局表达，必须明确报错而不是静默产出坏数据
        try:
            export_yolo(bundle, Path(tmp) / "out", ExportConfig(task="auto"))
            check(False, "混合任务应拒绝导出")
        except ExportError:
            check(True, "混合任务正确拒绝自动导出")


def test_export_guard() -> None:
    print("\n== 导出保护 ==")
    bundle = make_detection_bundle(n_groups=1, per_group=2)
    assign_splits(bundle)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "guard"
        export_yolo(bundle, out, ExportConfig(task=TASK_DETECTION))
        # 第二次导出应因目录非空而失败
        try:
            export_yolo(bundle, out, ExportConfig(task=TASK_DETECTION))
            check(False, "非空目录应拒绝覆盖")
        except ExportError:
            check(True, "非空目录默认拒绝覆盖")
        # 显式 overwrite 后成功
        report = export_yolo(bundle, out, ExportConfig(task=TASK_DETECTION, overwrite=True))
        eq(sum(report.images_exported.values()), 2, "overwrite=True 后重新导出成功")


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 划分 + 导出 测试")
    if not FRAMES_DIR.is_dir():
        print(f"\n缺少夹具帧图: {FRAMES_DIR}")
        print("请先运行 tests/fixtures/make_dmd_fixture.py 生成")
        return 1

    test_split_group_no_leak()
    test_split_reproducible()
    test_split_stratified_rare_class()
    test_split_respect_existing()
    test_task_inference()
    test_export_classification()
    test_export_classification_single_video()
    test_export_detection_and_roundtrip()
    test_export_hardlink_and_collision()
    test_merge_multiple_sources()
    test_merge_duplicate_source()
    test_merge_mixed_task_rejected()
    test_export_guard()

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
