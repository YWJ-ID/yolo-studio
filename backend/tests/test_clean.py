"""清洗模块测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_clean.py

覆盖：
    - 各类清洗规则的检出能力
    - dry-run 不改数据；apply 才落盘
    - 跨子集重复被识别为泄漏，且保留 train 侧
    - 近似重复的多重索引加速算法与暴力算法结果一致（不丢对）
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image, ImageDraw  # noqa: E402

from core.clean import (  # noqa: E402
    ACTION_REMOVE_IMAGE,
    SEVERITY_ERROR,
    CleanConfig,
    clean,
    list_rules,
)
from core.ingest import load_dataset  # noqa: E402
from core.io import find_near_duplicates  # noqa: E402
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
# 构造数据
# ---------------------------------------------------------------------------


def _make_image(path: Path, size=(40, 30), color=(10, 20, 30), marker: int = 0) -> None:
    """生成一张测试图。

    marker > 0 时画一个**位置随 marker 变化**的白色小方块。

    这一点很关键：只靠颜色差异区分图像是不可靠的——
    JPEG 会量化色差，相邻色号（如 RGB 的 (1,0,60) 与 (2,0,60)）
    编码后可能得到**完全相同的字节**，从而被去重规则误删。
    结构性差异才能稳定保证图片互不相同。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if marker:
        draw = ImageDraw.Draw(image)
        x = (marker * 7) % max(1, size[0] - 4)
        y = (marker * 5) % max(1, size[1] - 4)
        draw.rectangle([x, y, x + 3, y + 3], fill=(255, 255, 255))
    image.save(path, quality=90)


def make_bundle(tmp: Path) -> DatasetBundle:
    """构造一个"问题百出"的数据集，覆盖各类规则。"""
    img_dir = tmp / "imgs"
    bundle = DatasetBundle(source_id="messy", format_name="synthetic")
    seq = {"n": 0}

    def add(uid, name, w, h, split=None, group=None):
        seq["n"] += 1
        p = img_dir / name
        _make_image(p, (w, h), (20, 40, 60), marker=seq["n"])
        rec = ImageRecord(uid=uid, path=str(p), rel_path=name, width=w, height=h,
                          source_id="messy", split=split, group=group)
        bundle.add_image(rec)
        return rec

    # 1) 正常图
    normal = add("normal", "normal.jpg", 40, 30, SPLIT_TRAIN, "v1")
    bundle.add_annotation(Annotation("normal", "cat", BBox(5, 5, 20, 20)))

    # 2) 越界框 -> 应被裁剪
    oob = add("oob", "oob.jpg", 40, 30, SPLIT_TRAIN, "v1")
    bundle.add_annotation(Annotation("oob", "cat", BBox(35, 25, 60, 50)))

    # 3) 零面积框 -> 应被删除
    zero = add("zero", "zero.jpg", 40, 30, SPLIT_TRAIN, "v1")
    bundle.add_annotation(Annotation("zero", "cat", BBox(10, 10, 10, 20)))
    bundle.add_annotation(Annotation("zero", "cat", BBox(12, 12, 22, 22)))

    # 4) 重复标注（同图同类别同坐标）
    dup_ann = add("dupann", "dupann.jpg", 40, 30, SPLIT_TRAIN, "v1")
    bundle.add_annotation(Annotation("dupann", "cat", BBox(5, 5, 15, 15)))
    bundle.add_annotation(Annotation("dupann", "cat", BBox(5, 5, 15, 15)))

    # 5) 无标注图
    add("empty", "empty.jpg", 40, 30, SPLIT_TRAIN, "v1")

    # 6) 过小图
    add("small", "small.jpg", 8, 6, SPLIT_TRAIN, "v1")

    # 7) 跨子集完全重复：同一份文件内容分别放进 train 与 val
    #    marker=0 表示纯色无标记，与其它图（都有标记方块）字节不同
    dup_src = img_dir / "dup_a.jpg"
    _make_image(dup_src, (40, 30), (200, 10, 10))
    dup_b = img_dir / "dup_b.jpg"
    shutil.copyfile(dup_src, dup_b)
    for uid, name, split in (("dup_a", "dup_a.jpg", SPLIT_TRAIN), ("dup_b", "dup_b.jpg", SPLIT_VAL)):
        bundle.add_image(ImageRecord(uid=uid, path=str(img_dir / name), rel_path=name,
                                     width=40, height=30, source_id="messy",
                                     split=split, group="v_dup"))
        bundle.add_annotation(Annotation(uid, "cat", BBox(5, 5, 20, 20)))

    # 8) 同组跨子集（泄漏）
    same_g1 = add("sameg1", "sameg1.jpg", 40, 30, SPLIT_TRAIN, "video_leak")
    same_g2 = add("sameg2", "sameg2.jpg", 40, 30, SPLIT_TEST, "video_leak")
    for uid in ("sameg1", "sameg2"):
        bundle.add_annotation(Annotation(uid, "cat", BBox(5, 5, 20, 20)))

    # 9) 类别名变体 + 稀有类
    var = add("variant", "variant.jpg", 40, 30, SPLIT_TRAIN, "v1")
    bundle.add_annotation(Annotation("variant", "Closed Eye", BBox(5, 5, 15, 15)))
    var2 = add("variant2", "variant2.jpg", 40, 30, SPLIT_TRAIN, "v1")
    bundle.add_annotation(Annotation("variant2", "closed_eye", BBox(5, 5, 15, 15)))

    return bundle


# ---------------------------------------------------------------------------
# 规则检出
# ---------------------------------------------------------------------------


def test_rules_registered() -> None:
    print("\n== 规则注册 ==")
    rules = list_rules()
    ids = {r["id"] for r in rules}
    for expected in ("image_unreadable", "label_out_of_bounds", "leak_group_across_splits",
                     "category_name_variants", "class_too_few"):
        check(expected in ids, f"已注册规则 {expected}")
    eq(len(ids), len(rules), "规则 id 无重复")


def test_dry_run_no_mutation() -> None:
    print("\n== dry-run 不修改数据 ==")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = make_bundle(Path(tmp))
        before_images = len(bundle.images)
        before_anns = len(bundle.annotations)
        before_bbox = bundle.annotations_of("oob")[0].bbox.as_tuple()

        report = clean(bundle, CleanConfig())

        eq(len(bundle.images), before_images, "图像数未变")
        eq(len(bundle.annotations), before_anns, "标注数未变")
        eq(bundle.annotations_of("oob")[0].bbox.as_tuple(), before_bbox, "越界框未被修改")
        check(report.dry_run, "报告标记为 dry-run")
        check(len(report.findings) > 0, f"检出了 {len(report.findings)} 条问题")


def test_detection_of_each_rule() -> None:
    print("\n== 各规则检出 ==")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = make_bundle(Path(tmp))
        report = clean(bundle, CleanConfig())
        by_rule = report.counts_by_rule

        check(by_rule.get("label_out_of_bounds", 0) >= 1, "检出标注越界")
        check(by_rule.get("label_invalid_size", 0) >= 1, "检出零面积框")
        check(by_rule.get("label_duplicate", 0) >= 1, "检出重复标注")
        check(by_rule.get("image_without_annotation", 0) >= 1, "检出无标注图")
        check(by_rule.get("image_too_small", 0) >= 1, "检出过小图")
        check(by_rule.get("leak_group_across_splits", 0) >= 1, "检出同组跨子集")
        check(by_rule.get("category_name_variants", 0) >= 1, "检出类别名变体")
        check(by_rule.get("class_too_few", 0) >= 1, "检出样本过少的类别")
        check(by_rule.get("leak_duplicate_across_splits", 0) >= 1, "检出跨子集完全重复")

        # 泄漏必须是 error 级
        leaks = [f for f in report.findings if f.rule.startswith("leak_")]
        check(all(f.severity == SEVERITY_ERROR for f in leaks), "泄漏类问题均为 error 级")


def test_apply_clamps_and_removes() -> None:
    print("\n== apply 执行处置 ==")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = make_bundle(Path(tmp))
        images_before = len(bundle.images)
        anns_before = len(bundle.annotations)

        report = clean(bundle, CleanConfig(), apply=True)

        check(not report.dry_run, "报告标记为非 dry-run")
        check(report.images_after < images_before, "删除了问题图像")
        check(report.annotations_after < anns_before, "删除了问题标注")

        # 越界框被裁剪到图像范围内
        oob = bundle.images.get("oob")
        if oob is not None:
            ann = bundle.annotations_of("oob")[0]
            b = ann.bbox
            check(b.x2 <= 40 + 1e-6 and b.y2 <= 30 + 1e-6,
                  f"越界框已裁剪到范围内: {b.as_tuple()}")
        else:
            check(False, "oob 图像不应被删除（只裁剪）")

        # 零面积框被删除，但图像保留
        check("zero" in bundle.images, "零面积框所在图像保留")
        eq(len(bundle.annotations_of("zero")), 1, "零面积框已删除，保留有效框")

        # 重复标注只留一个
        eq(len(bundle.annotations_of("dupann")), 1, "重复标注已去重")

        # 跨子集重复：train 侧保留，val 侧删除
        check("dup_a" in bundle.images, "重复图保留 train 侧")
        check("dup_b" not in bundle.images, "删除 val 侧重复图")

        eq(report.actions_applied.get("clamp", 0) >= 1, True, "记录了裁剪操作")
        eq(report.actions_applied.get("remove_image", 0) >= 1, True, "记录了删图操作")


def test_apply_is_idempotent_ish() -> None:
    print("\n== 二次清洗应收敛 ==")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = make_bundle(Path(tmp))
        first = clean(bundle, CleanConfig(), apply=True)
        second = clean(bundle, CleanConfig(), apply=True)

        # 数据层面的问题应已消除
        for rule in ("label_out_of_bounds", "label_invalid_size",
                     "leak_duplicate_across_splits", "label_duplicate"):
            eq(second.counts_by_rule.get(rule, 0), 0, f"第二次清洗后 {rule} 已清零")

        # 但「同组跨子集」是划分问题，不是数据问题：
        # 正确解法是重新划分，而不是删数据，所以清洗只报告不处置。
        eq(second.counts_by_rule.get("leak_group_across_splits", 0), 1,
           "同组跨子集仍被报告（需靠重新划分解决，清洗不应删数据）")
        fixable = [f for f in second.findings if f.rule == "leak_group_across_splits"]
        eq(fixable[0].action, "report", "该规则的处置方式为仅报告")

        eq(second.images_after, first.images_after, "第二次未再删图")

        # 重新划分（开启分组防泄漏）后应消除
        assign_splits(bundle, SplitConfig(seed=1, respect_existing=False))
        third = clean(bundle, CleanConfig())
        eq(third.counts_by_rule.get("leak_group_across_splits", 0), 0,
           "重新划分后同组跨子集被消除")


def test_disabled_rules() -> None:
    print("\n== 规则开关 ==")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = make_bundle(Path(tmp))
        report = clean(bundle, CleanConfig(disabled_rules=["image_too_small", "class_too_few"]))
        eq(report.counts_by_rule.get("image_too_small", 0), 0, "被关闭的规则不产出结果")
        eq(report.counts_by_rule.get("class_too_few", 0), 0, "第二个被关闭的规则也不产出")
        check("image_too_small" in report.skipped_rules, "记录了被跳过的规则")
        check(report.counts_by_rule.get("label_out_of_bounds", 0) >= 1, "未关闭的规则照常运行")


def test_limit() -> None:
    print("\n== 限量预览 ==")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = make_bundle(Path(tmp))
        full = clean(bundle, CleanConfig())
        partial = clean(bundle, CleanConfig(limit=3))
        check(len(partial.findings) < len(full.findings), "limit 生效，检查范围缩小")


# ---------------------------------------------------------------------------
# 近似重复：加速算法正确性
# ---------------------------------------------------------------------------


def _brute_force(entries, max_distance):
    out = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i][1], entries[j][1]
            d = bin(int(a, 16) ^ int(b, 16)).count("1")
            if d <= max_distance:
                out.append((entries[i][0], entries[j][0]))
    return set(out)


def test_near_duplicate_algorithm() -> None:
    print("\n== 近似重复：加速算法 vs 暴力算法 ==")
    rng = random.Random(20240922)

    for trial, (n, max_distance) in enumerate([(200, 6), (400, 4), (300, 8)]):
        entries = []
        for i in range(n):
            # 一半随机，一半由已有哈希翻转少量位构造，制造真正的近邻
            if i > 20 and rng.random() < 0.5:
                base = int(entries[rng.randrange(i)][1], 16)
                flips = rng.randrange(0, max_distance + 1)
                for _ in range(flips):
                    base ^= 1 << rng.randrange(64)
                digest = f"{base:016x}"
            else:
                digest = f"{rng.getrandbits(64):016x}"
            entries.append((f"u{i}", digest))

        fast = set(find_near_duplicates(entries, max_distance))
        slow = _brute_force(entries, max_distance)

        eq(fast, slow, f"第 {trial + 1} 组 (n={n}, d={max_distance}) 与暴力结果完全一致")
        check(len(slow) > 0, f"第 {trial + 1} 组确实存在近邻对（{len(slow)} 对）")


def test_near_duplicate_scale() -> None:
    print("\n== 近似重复：规模可用性 ==")
    import time

    rng = random.Random(7)
    entries = [(f"u{i}", f"{rng.getrandbits(64):016x}") for i in range(20000)]
    started = time.perf_counter()
    pairs = find_near_duplicates(entries, 6)
    elapsed = time.perf_counter() - started
    check(elapsed < 20, f"2 万张图耗时 {elapsed:.2f}s（暴力法需 ~2 亿次比较）")
    check(isinstance(pairs, list), "返回结果类型正确")


# ---------------------------------------------------------------------------
# 真实夹具
# ---------------------------------------------------------------------------


def test_clean_real_fixture() -> None:
    print("\n== 真实夹具清洗 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return

    bundle = load_dataset(FIXTURE_DIR)
    assign_splits(bundle, SplitConfig(ratios=(0.7, 0.15, 0.15), seed=42))
    report = clean(bundle, CleanConfig(verify_readable=False))

    check(report.images_before == 60, "扫描到 60 帧")
    # 单视频夹具：所有帧同组，而划分不会拆散同组，因此不应报同组跨子集
    eq(report.counts_by_rule.get("leak_group_across_splits", 0), 0, "无同组跨子集泄漏")
    # 60 帧是同色块的渐变图，不应被判为完全重复
    eq(report.counts_by_rule.get("image_duplicate_exact", 0), 0, "无完全重复图")
    # drinking 恰好 5 个，阈值 5 时不应报（< 5 才算过少）
    eq(report.counts_by_rule.get("class_too_few", 0), 0, "阈值为 5 时三类都达标")

    strict = clean(bundle, CleanConfig(verify_readable=False, min_class_instances=10))
    check(strict.counts_by_rule.get("class_too_few", 0) >= 1,
          "把阈值提到 10 后，drinking(5) 被提示样本过少")


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 清洗模块测试")
    test_rules_registered()
    test_dry_run_no_mutation()
    test_detection_of_each_rule()
    test_apply_clamps_and_removes()
    test_apply_is_idempotent_ish()
    test_disabled_rules()
    test_limit()
    test_near_duplicate_algorithm()
    test_near_duplicate_scale()
    test_clean_real_fixture()

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
