"""M7 预标注测试（无需 pytest）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_prelabel.py

覆盖：
    * list_images：递归收集、单文件、目录不存在
    * build_bundle：图像数 / uid / 尺寸 / 相对路径
    * annotate_bundle：正常路径、boxes_total 累加、逐类计数、meta 血缘
    * 单张图推理失败（业务错误与子进程崩溃）→ 记入 failures 且不中断整批
    * 空目录行为
    * 分类模型分支（task=classify → kind=image，bbox=None）
    * run_prelabel 血缘写入 bundle.meta 与导出目录名前缀
    * 导出后 dataset_card.json 带上 prelabel 段

真实推理（.pt / .onnx）另外单独验证，不放进本文件。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image  # noqa: E402

from core.export import ExportConfig, export_yolo  # noqa: E402
from core.prelabel import (  # noqa: E402
    PRELABEL_PREFIX,
    SOURCE_MODEL,
    PrelabelConfig,
    annotate_bundle,
    build_bundle,
    ensure_prelabel_prefix,
    export_name,
    list_images,
    run_prelabel,
)

_failures: list = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [ok]   {message}")
    else:
        print(f"  [FAIL] {message}")
        _failures.append(message)


def eq(actual, expected, message: str) -> None:
    check(actual == expected, f"{message} (期望 {expected!r}, 实际 {actual!r})")


def contains(haystack: str, needle: str, message: str) -> None:
    check(needle in str(haystack), f"{message} (应在 {haystack!r} 中包含 {needle!r})")


# ---------------------------------------------------------------------------
# 假推理后端（不依赖 torch）：验证 IR 写入与会话交互
# ---------------------------------------------------------------------------

FAKE_WORKER_SCRIPT = '''
import json, os, sys

task = "detect"
names = []

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    t = msg.get("type")

    if t == "load":
        spec = msg.get("spec") or {}
        task = str(spec.get("task") or "detect").lower()
        names = list(spec.get("classes") or [])
        if not names:
            names = ["safe", "texting"] if task == "classify" else ["cat", "dog"]
        print(json.dumps({"type": "loaded", "ok": True, "task": task, "classes": names,
                          "num_classes": len(names), "weights": spec.get("weights"),
                          "format": spec.get("fmt", ""), "imgsz": spec.get("imgsz", 640),
                          "device": spec.get("device", "cpu")}), flush=True)

    elif t == "frame":
        fid = msg.get("id")
        name = os.path.basename(str(msg.get("path") or ""))
        if name == "boom.jpg":
            # 业务错误：worker 正常回包但结果 ok=false（如图像不可解码）
            print(json.dumps({"type": "result", "id": fid, "ok": False, "error": "解码失败: boom"}), flush=True)
            continue
        if name == "crash.jpg":
            sys.exit(3)
        if task == "classify":
            print(json.dumps({"type": "result", "id": fid, "ok": True, "task": "classify",
                              "width": 100, "height": 80, "class_names": names,
                              "top1": {"class_index": 0, "class_name": names[0], "confidence": 0.9},
                              "topk": []}), flush=True)
            continue
        dets = []
        if name != "empty.jpg":
            dets = [
                {"class_index": 0, "class_name": names[0], "confidence": 0.9, "bbox": [1, 2, 30, 40]},
                {"class_index": 1, "class_name": names[1], "confidence": 0.5, "bbox": [5, 6, 50, 60]},
            ]
        print(json.dumps({"type": "result", "id": fid, "ok": True, "task": "detect",
                          "width": 100, "height": 80, "class_names": names,
                          "detections": dets}), flush=True)

    elif t == "close":
        print(json.dumps({"type": "bye"}), flush=True)
        break
'''


def _fake_session(tmp: Path):
    """构造一个用假 worker 脚本的 InferSession（不加载 torch）。"""
    import subprocess
    import threading

    from core.infer.session import InferSession

    script = tmp / "fake_prelabel_worker.py"
    script.write_text(FAKE_WORKER_SCRIPT, encoding="utf-8")
    (tmp / "fake.pt").write_bytes(b"fake-pt")

    session = InferSession(python=sys.executable)

    def _start() -> None:
        session.proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        session._closing = False
        session._loaded = None
        session._results.clear()
        session.last_error = ""
        session._reader = threading.Thread(target=session._read_stdout, daemon=True)
        session._reader.start()
        session._stderr_reader = threading.Thread(target=session._read_stderr, daemon=True)
        session._stderr_reader.start()

    session.start = _start  # type: ignore[method-assign]
    return session


def _weights(tmp: Path) -> str:
    return str(tmp / "fake.pt")


def _make_images(root: Path, names, size=(100, 80)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", size, (i * 30 % 256, 40, 90)).save(root / name)
    return root


# ---------------------------------------------------------------------------
# list_images
# ---------------------------------------------------------------------------


def test_list_images() -> None:
    print("\n== list_images ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_images(root, ["a.jpg", "b.png"])
        sub = root / "sub"
        _make_images(sub, ["c.jpeg"])
        (root / "notes.txt").write_text("x", encoding="utf-8")
        # 非图片的伪装文件（扩展名像图片但不是）
        (root / "d.bmp").write_bytes(b"not-an-image")

        found = list_images(root)
        names = sorted(p.name for p in found)
        eq(names, ["a.jpg", "b.png", "c.jpeg", "d.bmp"], "递归收集所有图片扩展名")
        check(all(p.is_absolute() for p in found), "返回的是绝对路径")

        # 单文件
        one = list_images(root / "a.jpg")
        eq(len(one), 1, "传单个文件时只返回它")
        eq(list_images(root / "notes.txt"), [], "传非图片文件返回空")

        # 目录不存在
        try:
            list_images(root / "nope")
            check(False, "目录不存在应抛出 FileNotFoundError")
        except FileNotFoundError as exc:
            contains(str(exc), "不存在", "缺失目录报错明确")


# ---------------------------------------------------------------------------
# build_bundle
# ---------------------------------------------------------------------------


def test_build_bundle() -> None:
    print("\n== build_bundle ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_images(root, ["a.jpg"], size=(120, 90))
        _make_images(root / "sub", ["b.jpg"], size=(64, 48))

        images = list_images(root)
        bundle = build_bundle(images, PrelabelConfig(source_id="pl"))
        eq(len(bundle.images), 2, "图像已装入 IR")
        eq(bundle.format_name, "prelabel", "格式标记为 prelabel")
        check(all(im.uid for im in bundle.images.values()), "每张图都有 uid")
        eq(len({im.uid for im in bundle.images.values()}), 2, "uid 互不相同")
        eq(len(bundle.annotations), 0, "build_bundle 阶段还没有标注")

        by_name = {Path(im.path).name: im for im in bundle.images.values()}
        eq((by_name["a.jpg"].width, by_name["a.jpg"].height), (120, 90), "a.jpg 尺寸")
        eq((by_name["b.jpg"].width, by_name["b.jpg"].height), (64, 48), "b.jpg 尺寸")
        eq(
            Path(by_name["b.jpg"].rel_path).as_posix(),
            "sub/b.jpg",
            "相对路径保留子目录",
        )

        # 同样的输入 → 同样的 uid（稳定，便于重复导入去重）
        again = build_bundle(images, PrelabelConfig(source_id="pl"))
        eq(
            sorted(im.uid for im in bundle.images.values()),
            sorted(im.uid for im in again.images.values()),
            "uid 稳定可复现",
        )
        # 换 source_id → uid 不同
        other = build_bundle(images, PrelabelConfig(source_id="other"))
        check(
            set(im.uid for im in bundle.images.values())
            != set(im.uid for im in other.images.values()),
            "不同来源的 uid 不同",
        )


# ---------------------------------------------------------------------------
# annotate_bundle：检测
# ---------------------------------------------------------------------------


def test_annotate_detect() -> None:
    print("\n== annotate_bundle：检测正常路径 ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_images(root, ["a.jpg", "b.jpg", "empty.jpg"])
        session = _fake_session(root)
        try:
            bundle = build_bundle(list_images(root), PrelabelConfig(source_id="pl"))
            cfg = PrelabelConfig(
                weights=_weights(root), conf=0.3, iou=0.6, classes=["cat", "dog"]
            )
            report = annotate_bundle(bundle, cfg, session=session)

            eq(report.ok, True, "报告 ok")
            eq(report.images_total, 3, "图像总数")
            # 回归：boxes_total 必须累计（曾因未累加而报告 0 框、实际 21 个）
            eq(report.boxes_total, 4, "boxes_total 累计为 4（2+2+0）")
            eq(report.images_with_boxes, 2, "有标注的图 2 张")
            eq(report.images_empty, 1, "无标注的图 1 张")
            eq(report.images_failed, 0, "没有失败")
            eq(report.count_by_category, {"cat": 2, "dog": 2}, "逐类实例数")
            eq(report.classes, ["cat", "dog"], "报告记录模型类别")
            eq(len(bundle.annotations), 4, "IR 里确实写入了 4 条标注")

            ann = bundle.annotations[0]
            eq(ann.kind, "bbox", "标注形态为检测框")
            eq(ann.meta.get("source"), SOURCE_MODEL, "标注来源标记为模型预测")
            eq(ann.meta.get("weights"), _weights(root), "标注 meta 记录权重")
            eq(ann.meta.get("conf"), 0.3, "标注 meta 记录 conf")
            eq(ann.score, 0.9, "保留模型置信度 score")
            check(ann.bbox is not None, "检测标注带 bbox")

            # 空标注的图确实没有标注
            empty_im = next(im for im in bundle.images.values() if im.path.endswith("empty.jpg"))
            eq(bundle.annotations_of(empty_im.uid), [], "empty.jpg 无标注")

            eq(report.to_dict()["source"], SOURCE_MODEL, "to_dict 带来源标识")
            contains(report.to_dict()["disclaimer"], "人工复核", "to_dict 带免责说明")
        finally:
            session.stop()


# ---------------------------------------------------------------------------
# annotate_bundle：失败单张图不中断整批
# ---------------------------------------------------------------------------


def test_annotate_business_failure() -> None:
    print("\n== annotate_bundle：单张图业务失败 ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_images(root, ["a.jpg", "boom.jpg", "b.jpg"])
        session = _fake_session(root)
        try:
            bundle = build_bundle(list_images(root), PrelabelConfig(source_id="pl"))
            cfg = PrelabelConfig(weights=_weights(root), classes=["cat", "dog"])
            report = annotate_bundle(bundle, cfg, session=session)

            eq(report.ok, True, "个别图片失败不影响整批 ok")
            eq(report.images_failed, 1, "记录 1 张失败")
            eq(len(report.failures), 1, "failures 记录明细")
            contains(report.failures[0]["path"], "boom.jpg", "失败明细指向该图")
            contains(report.failures[0]["error"], "解码失败", "失败原因透传")
            # 关键：失败之后 b.jpg 仍然被推理，整批没有中断
            eq(report.images_with_boxes, 2, "失败前后的图都正常打标")
            eq(report.boxes_total, 4, "总框数只统计成功的图")
        finally:
            session.stop()


def test_annotate_infer_error() -> None:
    print("\n== annotate_bundle：子进程崩溃（InferError）==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_images(root, ["a.jpg", "crash.jpg"])
        session = _fake_session(root)
        try:
            bundle = build_bundle(list_images(root), PrelabelConfig(source_id="pl"))
            cfg = PrelabelConfig(weights=_weights(root), classes=["cat", "dog"])
            # 崩溃的图排在最后，验证异常被捕获并记入 failures，而不是抛出去
            report = annotate_bundle(bundle, cfg, session=session)

            eq(report.images_failed, 1, "崩溃被记为 1 张失败")
            contains(report.failures[0]["error"], "子进程", "失败原因是子进程退出")
            eq(report.images_with_boxes, 1, "崩溃前的图已正常打标")
        finally:
            session.stop()


# ---------------------------------------------------------------------------
# 空目录
# ---------------------------------------------------------------------------


def test_empty_dir() -> None:
    print("\n== 空目录 ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "empty").mkdir()
        eq(list_images(root / "empty"), [], "空目录收集到 0 张图")

        with tempfile.TemporaryDirectory() as tmp2:
            session = _fake_session(Path(tmp2))
            try:
                bundle, report = run_prelabel(
                    root / "empty",
                    PrelabelConfig(weights=str(Path(tmp2) / "fake.pt"), classes=["cat"]),
                    session=session,
                )
                eq(report.ok, False, "没有图片时报告为失败")
                contains(report.error, "没有", "给出明确原因")
                eq(report.images_total, 0, "图像总数为 0")
                eq(report.boxes_total, 0, "框数为 0")
                eq(len(bundle.images), 0, "IR 中没有图像")
            finally:
                session.stop()


# ---------------------------------------------------------------------------
# 分类模型分支
# ---------------------------------------------------------------------------


def test_classify_branch() -> None:
    print("\n== annotate_bundle：分类模型 ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_images(root, ["a.jpg", "b.jpg"])
        session = _fake_session(root)
        try:
            bundle = build_bundle(list_images(root), PrelabelConfig(source_id="pl"))
            cfg = PrelabelConfig(weights=_weights(root), task="classify")
            report = annotate_bundle(bundle, cfg, session=session)

            eq(report.task, "classify", "报告任务类型为分类")
            eq(report.boxes_total, 2, "分类时按整图标注计数")
            eq(report.images_with_boxes, 2, "两张图都有分类标注")
            eq(report.count_by_category, {"safe": 2}, "分类逐类计数")
            check(all(a.kind == "image" for a in bundle.annotations), "标注形态为 image")
            check(all(a.bbox is None for a in bundle.annotations), "分类标注 bbox 为 None")
            eq(bundle.annotations[0].category, "safe", "分类取 top1 类名")
            eq(bundle.annotations[0].score, 0.9, "分类保留 top1 置信度")
            contains(report.summary, "图像标注", "分类时 summary 用「标注」而非「框」")
        finally:
            session.stop()


# ---------------------------------------------------------------------------
# 血缘与命名
# ---------------------------------------------------------------------------


def test_lineage_and_naming() -> None:
    print("\n== 血缘与导出命名 ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "imgs"
        _make_images(root, ["a.jpg", "b.jpg"])
        session = _fake_session(tmp)
        try:
            cfg = PrelabelConfig(weights=str(tmp / "fake.pt"), conf=0.4, iou=0.55, classes=["cat", "dog"])
            bundle, report = run_prelabel(root, cfg, session=session)

            lineage = bundle.meta.get("prelabel")
            check(isinstance(lineage, dict), "run_prelabel 写入 bundle.meta['prelabel']")
            eq(lineage["weights"], cfg.weights, "血缘记录权重")
            eq(lineage["conf"], 0.4, "血缘记录 conf")
            eq(lineage["iou"], 0.55, "血缘记录 iou")
            eq(lineage["classes"], ["cat", "dog"], "血缘记录模型类别")
            check(bool(lineage["created_at"]), "血缘记录生成时间")
            eq(lineage["source"], SOURCE_MODEL, "血缘标明来源是模型预测")
            contains(lineage["disclaimer"], "人工复核", "血缘带免责说明")
            eq(report.to_dict() == lineage, True, "血缘与报告一致")
        finally:
            session.stop()

    eq(export_name("tiny").startswith(f"{PRELABEL_PREFIX}_"), True, "自动目录名带 prelabel 前缀")
    contains(export_name("tiny"), "tiny", "自动目录名保留来源名")
    eq(ensure_prelabel_prefix("myds"), "prelabel_myds", "无前缀时自动补上")
    eq(ensure_prelabel_prefix("prelabel_myds"), "prelabel_myds", "已有前缀不重复添加")
    check(ensure_prelabel_prefix("").startswith(f"{PRELABEL_PREFIX}_"), "空名走自动生成")


# ---------------------------------------------------------------------------
# 导出：dataset_card 写入 prelabel 段（M7-03）
# ---------------------------------------------------------------------------


def test_export_card() -> None:
    print("\n== 导出后 dataset_card 带 prelabel 段 ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "imgs"
        _make_images(root, ["a.jpg", "b.jpg"])
        session = _fake_session(tmp)
        try:
            cfg = PrelabelConfig(weights=str(tmp / "fake.pt"), conf=0.35, classes=["cat", "dog"])
            bundle, report = run_prelabel(root, cfg, session=session)
        finally:
            session.stop()

        out = tmp / "prelabel_out"
        result = export_yolo(bundle, out, ExportConfig())
        card = json.loads((out / "dataset_card.json").read_text(encoding="utf-8"))

        check("prelabel" in card, "dataset_card 固定带 prelabel 段")
        eq(card["prelabel"]["weights"], cfg.weights, "card 记录权重")
        eq(card["prelabel"]["conf"], 0.35, "card 记录 conf")
        eq(card["prelabel"]["classes"], ["cat", "dog"], "card 记录模型类别")
        eq(card["prelabel"]["source"], SOURCE_MODEL, "card 标明来源是模型预测")
        contains(card["prelabel"]["disclaimer"], "人工复核", "card 带免责说明")
        eq(result.task, "detection", "预标注结果导出为检测数据集")

        # 非预标注数据集：键仍在，值为 null
        plain = build_bundle(list_images(root), PrelabelConfig(source_id="plain"))
        out2 = tmp / "plain_out"
        export_yolo(plain, out2, ExportConfig(task="detection"))
        card2 = json.loads((out2 / "dataset_card.json").read_text(encoding="utf-8"))
        check("prelabel" in card2 and card2["prelabel"] is None, "非预标注数据集 prelabel 为 null")


def test_api_surface() -> None:
    print("\n== API：路由与错误路径（不加载模型）==")
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        check(False, f"TestClient 不可用: {exc}")
        return

    from app.main import app

    client = TestClient(app)
    eq(client.get("/api/prelabel/formats").status_code, 200, "GET /api/prelabel/formats 200")
    eq(client.get("/api/prelabel/weights").status_code, 200, "GET /api/prelabel/weights 200")

    jobs = client.get("/api/prelabel/jobs").json()
    check(isinstance(jobs.get("jobs"), list), "GET /api/prelabel/jobs 返回列表")

    base = Path(__file__).resolve().parent
    resp = client.post(
        "/api/prelabel/jobs",
        json={"weights": "x.pt", "images_dir": str(base / "nope_dir_xxx")},
    )
    eq(resp.status_code, 404, "图片目录不存在 404")

    resp = client.post(
        "/api/prelabel/jobs",
        json={"weights": str(base / "nope.pt"), "images_dir": str(base)},
    )
    eq(resp.status_code, 404, "权重不存在 404")

    resp = client.post("/api/prelabel/jobs", json={"weights": "x.pt", "images_dir": ""})
    check(resp.status_code in (400, 422), "缺 images_dir 报 4xx")

    eq(client.get("/api/prelabel/jobs/pl_nope").status_code, 404, "任务不存在 404")
    eq(client.get("/api/prelabel/jobs/pl_nope/samples").status_code, 404, "样本任务不存在 404")
    eq(client.delete("/api/prelabel/jobs/pl_nope").status_code, 404, "删除不存在的任务 404")


def test_export_surface() -> None:
    import core.prelabel as mod

    for name in (
        "PrelabelConfig",
        "PrelabelReport",
        "annotate_bundle",
        "build_bundle",
        "list_images",
        "run_prelabel",
        "ensure_prelabel_prefix",
        "export_name",
    ):
        check(hasattr(mod, name), f"core.prelabel 导出 {name}")


def main() -> int:
    print("== M7 预标注测试 ==")
    test_list_images()
    test_build_bundle()
    test_annotate_detect()
    test_annotate_business_failure()
    test_annotate_infer_error()
    test_empty_dir()
    test_classify_branch()
    test_lineage_and_naming()
    test_export_card()
    test_api_surface()
    test_export_surface()

    print("")
    if _failures:
        print(f"失败 {len(_failures)} 项:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
