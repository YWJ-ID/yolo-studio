"""类别规范化测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_taxonomy.py

覆盖：
    - 同义类名建议
    - 类名合法性校验
    - 合并 / 保留 / 丢弃 / 重排
    - **按标注形态过滤 —— 混合数据集（PROGRESS R-09）的解法**
    - 与导出模块联动：过滤后能正常导出为单一任务
"""

from __future__ import annotations

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
    export_yolo,
)
from core.ingest import load_dataset  # noqa: E402
from core.ir import (  # noqa: E402
    KIND_BBOX,
    KIND_IMAGE,
    SPLIT_TRAIN,
    Annotation,
    BBox,
    DatasetBundle,
    ImageRecord,
)
from core.split import SplitConfig, assign_splits  # noqa: E402
from core.taxonomy import (  # noqa: E402
    TaxonomyConfig,
    apply_taxonomy,
    normalize_key,
    sanitize_class_name,
    suggest_merges,
    validate_class_names,
)

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


def _frame(i: int) -> Path:
    return sorted(FRAMES_DIR.glob("*.jpg"))[i]


def make_mixed_bundle(n_det: int = 6, n_cls: int = 4) -> DatasetBundle:
    """构造检测框与图像级标注混合的数据集——这正是无法单布局导出的场景。"""
    bundle = DatasetBundle(source_id="mixed", format_name="synthetic")

    for i in range(n_det):
        uid = f"det_{i:03d}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(_frame(i)), rel_path=f"det_{i}.jpg",
                        width=48, height=36, source_id="mixed", split=SPLIT_TRAIN)
        )
        bundle.add_annotation(Annotation(uid, "cat", BBox(5, 5, 20, 20)))
        bundle.add_annotation(Annotation(uid, "dog", BBox(20, 10, 40, 30)))

    for i in range(n_cls):
        uid = f"cls_{i:03d}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(_frame(20 + i)), rel_path=f"cls_{i}.jpg",
                        width=48, height=36, source_id="mixed", split=SPLIT_TRAIN)
        )
        bundle.add_annotation(
            Annotation(uid, "walking", bbox=None, kind=KIND_IMAGE)
        )
    return bundle


def make_variant_bundle() -> DatasetBundle:
    """构造带同义类名变体的数据集。"""
    bundle = DatasetBundle(source_id="variants", format_name="synthetic")
    variants = [
        ("Closed Eye", 5),
        ("closed_eye", 3),
        ("closed-eye", 2),
    ]
    idx = 0
    for name, count in variants:
        for _ in range(count):
            uid = f"u{idx:03d}"
            bundle.add_image(
                ImageRecord(uid=uid, path=str(_frame(idx)), rel_path=f"u{idx}.jpg",
                            width=48, height=36, source_id="variants")
            )
            bundle.add_annotation(Annotation(uid, name, BBox(5, 5, 20, 20)))
            idx += 1
    # 再加一个正常类别
    uid = f"u{idx:03d}"
    bundle.add_image(
        ImageRecord(uid=uid, path=str(_frame(idx)), rel_path=f"u{idx}.jpg",
                    width=48, height=36, source_id="variants")
    )
    bundle.add_annotation(Annotation(uid, "Phone", BBox(5, 5, 20, 20)))
    return bundle


# ---------------------------------------------------------------------------
# 建议与校验
# ---------------------------------------------------------------------------


def test_normalize_key() -> None:
    print("\n== 类名归一化键 ==")
    eq(normalize_key("Closed Eye"), "closedeye", "空格")
    eq(normalize_key("closed_eye"), "closedeye", "下划线")
    eq(normalize_key("close-eyes"), "closeeyes", "连字符")
    eq(normalize_key("ClosedEye"), "closedeye", "大小写")
    check(normalize_key("Closed Eye") != normalize_key("Open Eye"), "不同类不会混淆")


def test_suggest_merges() -> None:
    print("\n== 同义类名建议 ==")
    bundle = make_variant_bundle()
    suggestions = suggest_merges(bundle)

    eq(len(suggestions), 1, "只发现一组疑似同义")
    group = suggestions[0]
    eq(set(group["names"]), {"Closed Eye", "closed_eye", "closed-eye"}, "组内类名")
    eq(group["suggested"], "Closed Eye", "建议规范名取实例数最多的写法")
    eq(group["total"], 10, "合并后总实例数")
    eq(group["counts"]["Closed Eye"], 5, "各类实例数")


def test_normalization_is_conservative() -> None:
    """归一化只处理标点/大小写差异，**不猜拼写错误**。

    这是有意的取舍：错误合并（把两个不同类合成一类）比漏合并危害大得多，
    它会直接污染类别语义且难以察觉。因此拼写差异留给人工补充映射。
    """
    print("\n== 归一化是保守的（不猜拼写）==")
    eq(normalize_key("close-eyes"), "closeeyes", "少了 d 的词归一化后不同")
    check(
        normalize_key("close-eyes") != normalize_key("Closed Eye"),
        "拼写差异不会被自动视为同类",
    )

    bundle = DatasetBundle(source_id="typo", format_name="synthetic")
    for i, name in enumerate(["Closed Eye", "close-eyes"]):
        uid = f"t{i}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(_frame(i)), rel_path=f"t{i}.jpg",
                        width=48, height=36, source_id="typo")
        )
        bundle.add_annotation(Annotation(uid, name, BBox(5, 5, 20, 20)))

    eq(suggest_merges(bundle), [], "不会为拼写差异给出合并建议")
    # 但人工可以在界面上显式指定映射
    report = apply_taxonomy(bundle, TaxonomyConfig(mapping={"close-eyes": "Closed Eye"}))
    eq(report.classes_after, ["Closed Eye"], "人工指定映射可以合并")
    eq(bundle.count_by_category(), {"Closed Eye": 2}, "实例数累加")


def test_validate_names() -> None:
    print("\n== 类名合法性校验 ==")
    issues = validate_class_names(["ok_name", ""])
    check(any(i["level"] == "error" and "空" in i["message"] for i in issues), "空类名报 error")

    issues = validate_class_names([r"bad/name", "bad\\name", "bad:name"])
    errors = [i for i in issues if i["level"] == "error"]
    check(len(errors) >= 3, f"非法文件名字符报 error（{len(errors)} 条）")

    issues = validate_class_names(["."])
    check(any("不能是 . " in i["message"] or "不能是 ." in i["message"] for i in issues),
          "'.' 被拒绝")

    issues = validate_class_names(["Closed Eye", "closed_eye"])
    check(any("归一化后相同" in i["message"] for i in issues), "归一化重复被提示")

    issues = validate_class_names([" padded "])
    check(any("空白" in i["message"] for i in issues), "首尾空白被提示")

    eq(validate_class_names(["good", "another"]), [], "合法类名无问题")


def test_sanitize() -> None:
    print("\n== 类名改造 ==")
    eq(sanitize_class_name("driver_actions/safe_drive"), "driver_actions_safe_drive", "斜杠被替换")
    eq(sanitize_class_name("a:b\\c"), "a_b_c", "多个非法字符")
    eq(sanitize_class_name("   "), "class", "空名兜底")
    eq(sanitize_class_name("normal_name"), "normal_name", "合法名不变")


# ---------------------------------------------------------------------------
# 合并 / 取舍 / 重排
# ---------------------------------------------------------------------------


def test_apply_mapping() -> None:
    print("\n== 类别合并 ==")
    bundle = make_variant_bundle()
    report = apply_taxonomy(
        bundle,
        TaxonomyConfig(mapping={"closed_eye": "Closed Eye", "closed-eye": "Closed Eye"}),
    )

    eq(report.classes_before, ["Closed Eye", "closed_eye", "closed-eye", "Phone"], "合并前类别")
    eq(report.classes_after, ["Closed Eye", "Phone"], "合并后类别")
    eq(report.merged, {"closed_eye": "Closed Eye", "closed-eye": "Closed Eye"}, "记录了合并关系")
    eq(report.removed_annotations, 0, "合并不删除标注")

    counts = bundle.count_by_category()
    eq(counts.get("Closed Eye"), 10, "合并后该类实例数累加")
    eq(counts.get("Phone"), 1, "其它类别不受影响")
    check(all(a.category in ("Closed Eye", "Phone") for a in bundle.annotations), "标注已重命名")


def test_drop_and_keep() -> None:
    print("\n== 类别丢弃 / 保留 ==")
    bundle = make_mixed_bundle()
    report = apply_taxonomy(bundle, TaxonomyConfig(drop_classes=["walking"]))
    eq(report.dropped_classes, ["walking"], "记录了被丢弃的类别")
    eq(report.removed_by_class, 4, "删除了 4 条 walking 标注")
    eq(report.removed_images, 4, "只剩图像级标注的 4 张图被一并删除")
    eq(bundle.category_names(), ["cat", "dog"], "类别表已收窄")

    bundle2 = make_mixed_bundle()
    report2 = apply_taxonomy(bundle2, TaxonomyConfig(keep_classes=["cat"]))
    eq(bundle2.category_names(), ["cat"], "只保留 cat")
    eq(report2.removed_by_class, 6 + 4, "dog 与 walking 的标注都被删除")


def test_class_order() -> None:
    print("\n== 类别顺序 ==")
    bundle = make_mixed_bundle()
    apply_taxonomy(bundle, TaxonomyConfig(class_order=["dog", "cat", "walking"]))
    eq(bundle.category_names(), ["dog", "cat", "walking"], "按指定顺序重排")

    bundle2 = make_mixed_bundle()
    report = apply_taxonomy(bundle2, TaxonomyConfig(class_order=["dog", "not_exist"]))
    eq(bundle2.category_names(), ["dog", "cat", "walking"], "不存在的类别被忽略并在末尾补齐")
    check(any("从未出现" in w for w in report.warnings), "对不存在的类别给出警告")


# ---------------------------------------------------------------------------
# 形态过滤：R-09 的解法
# ---------------------------------------------------------------------------


def test_kind_filter() -> None:
    print("\n== 按标注形态过滤（R-09）==")
    bundle = make_mixed_bundle()
    eq(bundle.stats()["annotation_kind"], "mixed", "过滤前是混合形态")

    report = apply_taxonomy(bundle, TaxonomyConfig(kind_filter=KIND_BBOX))
    eq(bundle.stats()["annotation_kind"], KIND_BBOX, "过滤后为纯检测框")
    eq(report.removed_by_kind, 4, "移除了 4 条图像级标注")
    eq(report.removed_images, 4, "只剩图像级标注的图被移除")
    eq(len(bundle.images), 6, "剩余 6 张检测图")

    bundle2 = make_mixed_bundle()
    apply_taxonomy(bundle2, TaxonomyConfig(kind_filter=KIND_IMAGE))
    eq(bundle2.stats()["annotation_kind"], KIND_IMAGE, "过滤后为纯图像级")
    eq(len(bundle2.images), 4, "剩余 4 张分类图")

    # 非法值必须报错，不能静默放行
    try:
        apply_taxonomy(make_mixed_bundle(), TaxonomyConfig(kind_filter="raw"))
        check(False, "非法 kind_filter 应报错")
    except ValueError:
        check(True, "非法 kind_filter 正确报错")


def test_kind_filter_enables_export() -> None:
    print("\n== 过滤后即可导出（端到端）==")
    bundle = make_mixed_bundle()

    # 过滤前：混合形态，导出必须拒绝
    try:
        export_yolo(bundle, Path(tempfile.mkdtemp()) / "x", ExportConfig(task="auto"))
        check(False, "混合形态应拒绝自动导出")
    except ExportError:
        check(True, "混合形态正确拒绝自动导出（不产出坏数据）")

    # 只留检测框 -> 导出检测数据集
    bundle_det = make_mixed_bundle()
    apply_taxonomy(bundle_det, TaxonomyConfig(kind_filter=KIND_BBOX))
    assign_splits(bundle_det, SplitConfig(ratios=(1.0, 0.0, 0.0), seed=1, min_val=0, min_test=0))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "det"
        report = export_yolo(bundle_det, out, ExportConfig(task=TASK_DETECTION))
        eq(report.task, TASK_DETECTION, "导出为检测任务")
        eq(sum(report.images_exported.values()), 6, "导出 6 张图")
        eq(report.boxes_exported, 12, "导出 12 个框")
        check((out / "labels" / SPLIT_TRAIN).is_dir(), "检测布局正确")

    # 只留图像级 -> 导出分类数据集
    bundle_cls = make_mixed_bundle()
    apply_taxonomy(bundle_cls, TaxonomyConfig(kind_filter=KIND_IMAGE))
    assign_splits(bundle_cls, SplitConfig(ratios=(1.0, 0.0, 0.0), seed=1, min_val=0, min_test=0))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cls"
        report = export_yolo(bundle_cls, out, ExportConfig(task=TASK_CLASSIFICATION))
        eq(report.task, TASK_CLASSIFICATION, "导出为分类任务")
        eq(report.classes, ["walking"], "分类类别")
        check((out / SPLIT_TRAIN / "walking").is_dir(), "分类布局正确")


def test_sanitize_makes_names_exportable() -> None:
    print("\n== 含非法字符的类名经改造后可导出 ==")
    bundle = DatasetBundle(source_id="bad", format_name="synthetic")
    bundle.add_image(
        ImageRecord(uid="a", path=str(_frame(0)), rel_path="a.jpg",
                    width=48, height=36, source_id="bad", split=SPLIT_TRAIN)
    )
    bundle.add_annotation(
        Annotation("a", "driver_actions/safe_drive", bbox=None, kind=KIND_IMAGE)
    )

    issues = validate_class_names(bundle.category_names())
    check(any(i["level"] == "error" for i in issues), "未处理时类名不合法")

    apply_taxonomy(bundle, TaxonomyConfig(sanitize=True))
    eq(bundle.category_names(), ["driver_actions_safe_drive"], "类名已改造为安全形式")
    eq(validate_class_names(bundle.category_names()), [], "改造后无问题")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "safe"
        report = export_yolo(bundle, out, ExportConfig(task=TASK_CLASSIFICATION))
        check((out / SPLIT_TRAIN / "driver_actions_safe_drive").is_dir(), "分类目录正确落盘")


def test_empty_after_filter() -> None:
    print("\n== 过滤后为空的情形 ==")
    bundle = make_mixed_bundle()
    report = apply_taxonomy(bundle, TaxonomyConfig(keep_classes=["nonexistent"]))
    eq(bundle.category_names(), [], "类别清空")
    eq(len(bundle.annotations), 0, "标注清空")
    eq(len(bundle.images), 0, "图像清空")
    check(any("无法训练" in w for w in report.warnings), "明确警告数据集无法训练")


def test_real_fixture_suggestion() -> None:
    print("\n== 真实 DMD 夹具 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return
    bundle = load_dataset(FIXTURE_DIR)
    # safe_drive / texting_right / drinking 归一化后互不相同
    eq(suggest_merges(bundle), [], "无同义类别")
    eq(validate_class_names(bundle.category_names()), [], "类名全部合法")

    report = apply_taxonomy(bundle, TaxonomyConfig(class_order=["drinking", "safe_drive", "texting_right"]))
    eq(report.classes_after, ["drinking", "safe_drive", "texting_right"], "类别重排生效")
    eq(report.removed_annotations, 0, "重排不删标注")


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 类别规范化测试")
    if not FRAMES_DIR.is_dir():
        print(f"\n缺少夹具帧图: {FRAMES_DIR}")
        return 1

    test_normalize_key()
    test_suggest_merges()
    test_normalization_is_conservative()
    test_validate_names()
    test_sanitize()
    test_apply_mapping()
    test_drop_and_keep()
    test_class_order()
    test_kind_filter()
    test_kind_filter_enables_export()
    test_sanitize_makes_names_exportable()
    test_empty_after_filter()
    test_real_fixture_suggestion()

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
