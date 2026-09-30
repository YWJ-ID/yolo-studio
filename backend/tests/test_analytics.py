"""统计分析与质量报告测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_analytics.py
"""

from __future__ import annotations

import base64
import re
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.analytics import (  # noqa: E402
    AnalyticsConfig,
    analyze,
    hbar_chart,
    heat_cell,
    sample_images,
    vbar_chart,
    write_html_report,
)
from core.ingest import load_dataset  # noqa: E402
from core.ir import (  # noqa: E402
    KIND_BBOX,
    KIND_IMAGE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Annotation,
    BBox,
    DatasetBundle,
    ImageRecord,
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


def make_bundle() -> DatasetBundle:
    """构造分布已知的数据集，用于精确校验统计结果。

    图像声明为 640x480（分析用声明尺寸），但文件用夹具里的小图，
    这样缩略图仍能真实生成。
    """
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    bundle = DatasetBundle(source_id="analytics", format_name="synthetic", root="/demo/root")

    # 类别实例数：cat=6, dog=3, bird=1（明显不均衡，bird 属样本过少）
    plan = [
        ("a", "cat", [(0, 0, 20, 20)], SPLIT_TRAIN),        # 面积 400 -> 小
        ("b", "cat", [(0, 0, 40, 40)], SPLIT_TRAIN),        # 面积 1600 -> 中
        ("c", "cat", [(0, 0, 200, 200)], SPLIT_TRAIN),      # 面积 40000 -> 大
        ("d", "cat", [(0, 0, 20, 20)], SPLIT_VAL),
        ("e", "cat", [(0, 0, 20, 20)], SPLIT_TEST),
        ("f", "cat", [(0, 0, 20, 20)], SPLIT_TRAIN),
        ("g", "dog", [(0, 0, 30, 30)], SPLIT_TRAIN),
        ("h", "dog", [(0, 0, 30, 30)], SPLIT_VAL),
        ("i", "dog", [(0, 0, 30, 30)], SPLIT_TRAIN),
        ("j", "bird", [(0, 0, 20, 20)], SPLIT_TRAIN),       # 只出现在 train
    ]

    for idx, (key, cls, boxes, split) in enumerate(plan):
        uid = f"img_{key}"
        bundle.add_image(
            ImageRecord(
                uid=uid, path=str(frames[idx % len(frames)]), rel_path=f"{key}.jpg",
                width=640, height=480, source_id="analytics", split=split,
            )
        )
        for box in boxes:
            bundle.add_annotation(Annotation(uid, cls, BBox(*box), kind=KIND_BBOX))

    # 一张无标注图（检测任务中是合法背景样本）
    bundle.add_image(
        ImageRecord(uid="img_z", path=str(frames[0]), rel_path="z.jpg",
                    width=640, height=480, source_id="analytics", split=SPLIT_TRAIN)
    )
    return bundle


# 类别实例数最多只有 6，用默认阈值 20 会让三类都被判为"样本不足"，
# 无法区分。这里用 3 作为阈值，只有 bird(1) 会被标记。
TEST_CONFIG = AnalyticsConfig(min_class_instances=3)


def analyze_demo():
    return analyze(make_bundle(), TEST_CONFIG)


# ---------------------------------------------------------------------------
# 统计正确性
# ---------------------------------------------------------------------------


def test_summary_and_distribution() -> None:
    print("\n== 概览与类别分布 ==")
    report = analyze_demo()
    s = report.summary

    eq(s["num_images"], 11, "图像总数（10 张有标注 + 1 张背景）")
    eq(s["num_annotations"], 10, "标注总数")
    eq(s["num_classes"], 3, "类别数")
    eq(s["split_counts"], {SPLIT_TRAIN: 8, SPLIT_VAL: 2, SPLIT_TEST: 1}, "子集分布")

    dist = {c["name"]: c["count"] for c in report.class_distribution}
    eq(dist, {"cat": 6, "dog": 3, "bird": 1}, "各类实例数")
    # 分布按数量降序
    eq([c["name"] for c in report.class_distribution], ["cat", "dog", "bird"], "按数量降序排列")
    eq(report.class_distribution[0]["share"], 0.6, "占比计算")


def test_size_categories() -> None:
    print("\n== 目标尺寸分布 ==")
    report = analyze_demo()
    sizes = {x["name"][:1]: x["count"] for x in report.size_category}
    # 小目标: 4 个 cat 20x20(400) + 3 个 dog 30x30(900) + 1 个 bird 20x20 = 8
    # 中目标: 1 个 40x40(1600)；大目标: 1 个 200x200(40000)
    eq(sizes.get("小"), 8, "小目标数量（面积 <32²）")
    eq(sizes.get("中"), 1, "中目标数量（32²~96²）")
    eq(sizes.get("大"), 1, "大目标数量（>96²）")
    eq(sum(sizes.values()), 10, "三档之和等于检测框总数")


def test_objects_per_image_histogram() -> None:
    print("\n== 每图目标数 ==")
    report = analyze_demo()
    hist = report.objects_per_image_histogram
    # 10 张各有 1 个目标，1 张有 0 个 -> 首桶(0~1) 计 1，次桶(1~2) 计 10
    eq(hist["labels"][0], "0~1", "首桶边界从 0 开始")
    eq(hist["counts"][0], 1, "无标注图计入首桶（而非被误算到末桶）")
    eq(hist["counts"][1], 10, "单目标图数量")
    eq(sum(hist["counts"]), 11, "分桶总数等于图像总数")


def test_image_sizes() -> None:
    print("\n== 图像尺寸 ==")
    report = analyze_demo()
    eq(len(report.image_sizes), 1, "只有一种尺寸")
    eq(report.image_sizes[0]["size"], "640x480", "尺寸表示")
    eq(report.image_sizes[0]["count"], 11, "数量")
    check(any("尺寸一致" in f["message"] for f in report.findings), "对尺寸一致给出结论")


def test_class_by_split_matrix() -> None:
    print("\n== 类别 × 子集矩阵 ==")
    report = analyze_demo()
    eq(report.class_by_split[SPLIT_TRAIN]["cat"], 4, "train 的 cat 实例数")
    eq(report.class_by_split[SPLIT_VAL]["cat"], 1, "val 的 cat 实例数")
    eq(report.class_by_split[SPLIT_VAL]["bird"], 0, "val 没有 bird")
    eq(report.class_by_split[SPLIT_TEST]["dog"], 0, "test 没有 dog")

    # 缺失必须被明确报告（否则用户只看到 NaN 指标不知原因）
    messages = " ".join(f["message"] for f in report.findings)
    check("bird" in messages, "缺失类别被写入结论")
    eq(report.split_summary[0]["classes_present"], 3, "train 覆盖 3 个类别")
    eq(report.split_summary[1]["classes_present"], 2, "val 覆盖 cat 与 dog 两个类别")


def test_imbalance() -> None:
    print("\n== 不均衡量化 ==")
    report = analyze_demo()
    imb = report.imbalance
    eq(imb["max_class"], "cat", "最大类")
    eq(imb["min_class"], "bird", "最小类")
    eq(imb["ratio"], 6.0, "最大/最小比")
    eq([u["name"] for u in imb["underrepresented"]], ["bird"], "样本不足的类别")
    check(imb["top20pct_share"] > 0, f"头部集中度已计算: {imb['top20pct_share']}")


def test_findings_and_recommendations() -> None:
    print("\n== 结论与建议 ==")
    report = analyze_demo()
    check(len(report.findings) > 0, f"产出了 {len(report.findings)} 条结论")
    check(len(report.recommendations) > 0, f"产出了 {len(report.recommendations)} 条建议")
    check(
        any("背景负样本" in r for r in report.recommendations),
        "对检测任务的无标注图给出正确建议（可作背景负样本）",
    )
    check(
        any("样本量偏低" in r or "补充" in r for r in report.recommendations),
        "对样本不足的类别给出建议",
    )
    # 小目标占多数 -> 应提示提高 imgsz
    check(
        any("小目标" in r for r in report.recommendations),
        "小目标占多数时提示提高 imgsz",
    )


def test_split_distribution_shift() -> None:
    print("\n== 子集类别构成偏移 ==")
    # train 全是 cat，val 全是 dog：val 的构成与整体严重不符
    bundle = DatasetBundle(source_id="shift", format_name="synthetic")
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    plan = [("t1", "cat", SPLIT_TRAIN), ("t2", "cat", SPLIT_TRAIN),
            ("t3", "cat", SPLIT_TRAIN), ("v1", "dog", SPLIT_VAL)]
    for i, (key, cls, split) in enumerate(plan):
        uid = f"im_{key}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(frames[i]), rel_path=f"{key}.jpg",
                        width=640, height=480, source_id="shift", split=split)
        )
        bundle.add_annotation(Annotation(uid, cls, BBox(0, 0, 30, 30), kind=KIND_BBOX))

    report = analyze(bundle)
    messages = " ".join(f["message"] for f in report.findings)
    check("构成与整体不一致" in messages, "检出子集类别构成偏移")
    check(
        any("分层" in r for r in report.recommendations),
        "建议开启分层划分",
    )

    # 构成一致时不应误报
    balanced = DatasetBundle(source_id="bal", format_name="synthetic")
    for i in range(4):
        split = SPLIT_TRAIN if i < 2 else SPLIT_VAL
        uid = f"b{i}"
        balanced.add_image(
            ImageRecord(uid=uid, path=str(frames[i]), rel_path=f"b{i}.jpg",
                        width=640, height=480, source_id="bal", split=split)
        )
        balanced.add_annotation(Annotation(uid, "cat", BBox(0, 0, 30, 30), kind=KIND_BBOX))
    ok_report = analyze(balanced)
    check(
        "构成与整体不一致" not in " ".join(f["message"] for f in ok_report.findings),
        "构成一致时不误报",
    )


def test_classification_task_advice() -> None:
    print("\n== 分类任务的不同建议 ==")
    bundle = DatasetBundle(source_id="cls", format_name="synthetic")
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    for i in range(6):
        uid = f"c{i}"
        bundle.add_image(
            ImageRecord(uid=uid, path=str(frames[i]), rel_path=f"c{i}.jpg",
                        width=48, height=36, source_id="cls", split=SPLIT_TRAIN)
        )
        bundle.add_annotation(Annotation(uid, "walking", bbox=None, kind=KIND_IMAGE))
    # 一张无标签图，分类任务中应是 error
    bundle.add_image(
        ImageRecord(uid="c9", path=str(frames[10]), rel_path="c9.jpg",
                    width=48, height=36, source_id="cls", split=SPLIT_TRAIN)
    )

    report = analyze(bundle)
    levels = {f["level"] for f in report.findings}
    check("error" in levels, "分类任务中的无标签图被标为 error")
    check(
        any("无效样本" in f["message"] for f in report.findings),
        "明确指出分类任务下无标签图是无效样本",
    )


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------


def test_sample_images() -> None:
    print("\n== 抽样预览 ==")
    bundle = make_bundle()
    samples = sample_images(bundle, limit=3)
    eq(len(samples), 3, "按 limit 截断")

    cats = set()
    for s in samples:
        for o in s["objects"]:
            cats.add(o["category"])
    check(len(cats) >= 2, f"抽样覆盖多个类别（实际 {sorted(cats)}）")

    eq(sample_images(bundle, limit=0), [], "limit=0 返回空")
    big = sample_images(bundle, limit=100)
    eq(len(big), len(bundle.images), "limit 超过总数时返回全部，不重复")

    # 标注字段结构完整
    first = samples[0]
    for key in ("uid", "path", "width", "height", "split", "num_objects", "objects"):
        check(key in first, f"抽样字段 {key} 存在")


# ---------------------------------------------------------------------------
# 图表与 HTML
# ---------------------------------------------------------------------------


def test_charts() -> None:
    print("\n== SVG 图表 ==")
    svg = hbar_chart([("cat", 6), ("dog", 3)], title="类别")
    check(svg.startswith("<svg"), "水平条形图是合法 SVG")
    check("cat" in svg and "dog" in svg, "包含标签")

    svg2 = vbar_chart(["0~1", "1~2"], [1, 10], title="分布")
    check(svg2.startswith("<svg"), "垂直柱状图是合法 SVG")
    check("0~1" in svg2, "包含分桶标签")

    # 空数据不应崩溃
    check("<svg" in hbar_chart([]), "空数据返回占位 SVG")
    check("<svg" in vbar_chart([], []), "空分桶返回占位 SVG")

    # 标签需要转义，避免注入
    evil = hbar_chart([("<script>x</script>", 1)])
    check("<script>" not in evil, "标签中的 HTML 被转义")

    eq(heat_cell(0, 10).startswith("background:#fff1f0"), True, "0 值单元格用醒目标记")
    check("rgba" in heat_cell(8, 10), "非 0 值按比例着色")


def test_html_report() -> None:
    print("\n== HTML 报告 ==")
    report = analyze_demo()
    html = write_html_report(report, Path(tempfile.mkdtemp()) / "report.html", title="测试报告").read_text(
        encoding="utf-8"
    )

    check(html.startswith("<!doctype html>"), "是完整 HTML 文档")
    for section in ("检查结论", "处置建议", "类别分布", "类别 × 子集", "目标尺寸分布", "抽样预览"):
        check(section in html, f"包含章节「{section}」")

    check("测试报告" in html, "标题生效")
    check("<svg" in html, "内嵌 SVG 图表")
    # 缩略图以 base64 内嵌，报告不依赖外部文件
    check("data:image/jpeg;base64," in html, "缩略图以 base64 内嵌")
    check("http://" not in html.split("</style>")[1] or "https://" not in html, "不引用外部资源")

    # 内嵌图片能被真正解码（不是占位串）
    match = re.search(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", html)
    check(match is not None, "找到内嵌图片数据")
    if match:
        raw = base64.b64decode(match.group(1))
        eq(raw[:2], b"\xff\xd8", "内嵌数据是合法 JPEG 头")

    # 大图不应被完整内嵌（缩略图要真的缩小）
    check(len(raw) < 60_000, f"缩略图体积受控 ({len(raw)} bytes)")


def test_html_report_without_embedding() -> None:
    print("\n== 报告可选不内嵌图片 ==")
    report = analyze_demo()
    out = Path(tempfile.mkdtemp()) / "no_img.html"
    write_html_report(report, out, embed_samples=False)
    html = out.read_text(encoding="utf-8")
    check("data:image/jpeg" not in html, "关闭内嵌后不含图片数据")
    check("类别分布" in html, "其余章节仍然完整")


def test_real_fixture() -> None:
    print("\n== 真实 DMD 夹具 ==")
    if not FIXTURE_DIR.is_dir():
        check(False, "夹具不存在，跳过")
        return
    bundle = load_dataset(FIXTURE_DIR)
    report = analyze(bundle)

    eq(report.summary["num_images"], 60, "帧数")
    eq(report.summary["num_classes"], 3, "类别数")
    eq(report.summary["annotation_kind"], KIND_IMAGE, "形态为图像级")
    eq(len(report.size_category), 3, "尺寸分档存在（图像级标注无框，计数为 0）")
    check(all(x["count"] == 0 for x in report.size_category), "无检测框时尺寸分布为空")
    check(len(report.recommendations) > 0, "仍给出建议（如划分缺失）")

    html = write_html_report(report, Path(tempfile.mkdtemp()) / "dmd.html")
    check(html.stat().st_size > 5000, f"报告已生成 ({html.stat().st_size} bytes)")


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 统计分析测试")
    if not FRAMES_DIR.is_dir():
        print(f"\n缺少夹具帧图: {FRAMES_DIR}")
        return 1

    test_summary_and_distribution()
    test_size_categories()
    test_objects_per_image_histogram()
    test_image_sizes()
    test_class_by_split_matrix()
    test_imbalance()
    test_findings_and_recommendations()
    test_split_distribution_shift()
    test_classification_task_advice()
    test_sample_images()
    test_charts()
    test_html_report()
    test_html_report_without_embedding()
    test_real_fixture()

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
