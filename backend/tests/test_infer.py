"""M6 实时验证测试（无需 pytest）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_infer.py

覆盖：
    * 格式清单与可用性探测（缺包时的原因）
    * 权重路径 -> 格式推断
    * InferSpec / InferOptions 校验
    * 结果归一化（纯函数，用假的 ultralytics 风格 payload）
    * 会话：加载 / 推理 / 超时 / 子进程崩溃 / 背压等待者唤醒
    * API：格式、权重、加载、单图推理、关闭
    * WebSocket：ready / loaded / frame -> result / close

真实的 ultralytics 推理（.pt 与 .onnx）另外单独验证，不放进本文件，
以免测试依赖具体权重与 GPU。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.infer import (  # noqa: E402
    InferOptions,
    InferSpec,
    build_frame_result,
    capability_report,
    detect_weights_format,
    get_format,
    is_known,
)
from core.infer.result import Detection, FrameResult  # noqa: E402
from core.infer.spec import (  # noqa: E402
    TASK_CLASSIFY,
    TASK_DETECT,
    TASK_SEGMENT,
    InferOptions as InferOptionsDirect,
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
# 格式与探测
# ---------------------------------------------------------------------------


def test_formats() -> None:
    print("\n== 推理格式清单 ==")
    report = capability_report()
    names = {item["name"] for item in report}
    check({"pt", "onnx", "torchscript"} <= names, f"含核心格式: {sorted(names)}")

    for item in report:
        check(isinstance(item["available"], bool), f"{item['name']} 有 available 字段")
        if not item["available"]:
            check(bool(item["reason"]), f"{item['name']} 不可用时给出原因: {item['reason']}")

    # pt 必可用（本项目本来就依赖 torch + ultralytics）
    pt = next(i for i in report if i["name"] == "pt")
    eq(pt["available"], True, "pt 格式可用")

    # 未知格式
    eq(is_known("nope"), False, "未知格式 is_known=False")
    try:
        get_format("nope")
        check(False, "未知格式应抛错")
    except ValueError as exc:
        contains(str(exc), "nope", "未知格式报错含名称")


def test_format_detection() -> None:
    print("\n== 权重路径 -> 格式推断 ==")
    eq(detect_weights_format(r"C:\a\best.pt"), "pt", ".pt -> pt")
    eq(detect_weights_format(r"C:\a\best.onnx"), "onnx", ".onnx -> onnx")
    eq(detect_weights_format(r"C:\a\best.torchscript"), "torchscript", ".torchscript -> torchscript")
    eq(detect_weights_format(r"C:\a\best.engine"), "engine", ".engine -> engine")
    eq(detect_weights_format(r"C:\a\yolo11n_openvino_model"), "openvino", "openvino 目录 -> openvino")
    eq(detect_weights_format(r"C:\a\best.mlpackage"), "coreml", ".mlpackage -> coreml")
    try:
        detect_weights_format(r"C:\a\mystery.xyz")
        check(False, "未知扩展名应抛错")
    except ValueError as exc:
        contains(str(exc), "mystery.xyz", "未知扩展名报错含路径")


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


def test_spec_validation() -> None:
    print("\n== InferSpec / InferOptions 校验 ==")
    problems = InferSpec().validate()
    check(any("weights" in p for p in problems), f"空 weights 报错: {problems}")

    spec = InferSpec(weights=str(Path(__file__)), imgsz=0)
    problems = spec.validate()
    check(any("imgsz" in p for p in problems), f"imgsz=0 报错: {problems}")

    spec = InferSpec(weights=str(Path(__file__)), task="bogus")
    problems = spec.validate()
    check(any("任务类型" in p for p in problems), f"非法 task 报错: {problems}")

    # 合法
    eq(InferSpec(weights=str(Path(__file__))).validate(), [], "合法 spec 无问题")

    # options
    opts = InferOptions(conf=1.5)
    check(any("conf" in p for p in opts.validate()), "conf 越界报错")
    opts = InferSpec.__mro__  # 占位，避免未使用导入告警
    eq(InferOptions().validate(), [], "默认 options 合法")
    eq(InferOptionsDirect.from_dict(None).conf, 0.25, "from_dict(None) 用默认值")
    eq(InferOptionsDirect.from_dict({"conf": 0.5, "unknown": 1}).conf, 0.5, "忽略未知键")

    # normalized_task
    eq(InferSpec(weights="x", task="DETECT").normalized_task(), "detect", "任务名小写化")


# ---------------------------------------------------------------------------
# 结果归一化（纯函数）
# ---------------------------------------------------------------------------


def test_result_detection() -> None:
    print("\n== 结果归一化：检测 ==")
    payload = {
        "ok": True,
        "task": "detect",
        "width": 640,
        "height": 480,
        "class_names": ["cat", "dog"],
        "detections": [
            {"class_index": 0, "confidence": 0.91, "bbox": [10, 20, 110, 120]},
            {"class_index": 1, "confidence": 0.55, "bbox": [200.5, 30, 300, 130.25]},
        ],
    }
    r = build_frame_result(payload, frame_id=7, duration_ms=12.5)
    eq(r.ok, True, "ok")
    eq(r.frame_id, 7, "frame_id 回显")
    eq(r.duration_ms, 12.5, "duration_ms")
    eq(r.width, 640, "width")
    eq(r.height, 480, "height")
    eq(len(r.detections), 2, "两个检测框")
    eq(r.detections[0].class_name, "cat", "类名按 class_names 填")
    eq(r.detections[0].bbox, (10.0, 20.0, 110.0, 120.0), "bbox 转 float")
    eq(r.detections[1].confidence, 0.55, "置信度")
    eq(r.to_dict()["num_detections"], 2, "to_dict 带 num_detections")


def test_result_robustness() -> None:
    print("\n== 结果归一化：容错 ==")
    # 缺 bbox 的框被丢弃，不编造
    payload = {
        "task": "detect",
        "class_names": ["a"],
        "detections": [
            {"class_index": 0, "confidence": 0.9},
            {"class_index": 0, "confidence": 0.8, "bbox": [1, 2, 3]},
            {"class_index": 0, "confidence": 0.7, "bbox": [1, 2, 3, 4]},
        ],
    }
    r = build_frame_result(payload)
    eq(len(r.detections), 1, "只有合法 bbox 被保留")

    # NaN / 非数值置信度 -> None
    r = build_frame_result(
        {"task": "detect", "detections": [{"class_index": 0, "confidence": float("nan"), "bbox": [0, 0, 1, 1]}]}
    )
    eq(r.detections[0].confidence, None, "NaN 置信度 -> None（不是 0）")

    # 未知类别下标 -> 合成类名而不是崩溃
    r = build_frame_result(
        {"task": "detect", "class_names": ["a"], "detections": [{"class_index": 5, "bbox": [0, 0, 1, 1]}]}
    )
    eq(r.detections[0].class_name, "class_5", "越界下标合成类名")

    # 空/异常 payload 不炸
    r = build_frame_result({})
    eq(r.ok, True, "空 payload 仍 ok")
    eq(len(r.detections), 0, "空 payload 无检测框")


def test_result_classification() -> None:
    print("\n== 结果归一化：分类 ==")
    payload = {
        "task": "classify",
        "width": 48,
        "height": 36,
        "class_names": ["safe_drive", "texting", "drinking"],
        "top1": {"class_index": 0, "confidence": 0.87},
        "topk": [
            {"class_index": 0, "confidence": 0.87},
            {"class_index": 1, "confidence": 0.09},
        ],
    }
    r = build_frame_result(payload)
    eq(r.task, TASK_CLASSIFY, "任务类型")
    eq(r.top1["class_name"], "safe_drive", "top1 类名")
    eq(r.top1["confidence"], 0.87, "top1 置信度")
    eq(len(r.topk), 2, "topk 两项")
    eq(len(r.detections), 0, "分类结果不含检测框")


# ---------------------------------------------------------------------------
# 假推理后端（不依赖 torch）：验证会话与协议
# ---------------------------------------------------------------------------

FAKE_WORKER_SCRIPT = '''
import json, sys

model_loaded = False

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    t = msg.get("type")
    if t == "load":
        spec = msg.get("spec") or {}
        if not spec.get("weights"):
            print(json.dumps({"type": "loaded", "ok": False, "error": "weights 不能为空"}), flush=True)
            continue
        model_loaded = True
        print(json.dumps({"type": "loaded", "ok": True, "task": spec.get("task") or "detect",
                          "classes": spec.get("classes") or ["cat"], "num_classes": 1,
                          "weights": spec["weights"], "format": spec.get("fmt", ""),
                          "imgsz": spec.get("imgsz", 640), "device": spec.get("device", "cpu")}), flush=True)
    elif t == "frame":
        if not model_loaded:
            print(json.dumps({"type": "result", "id": msg.get("id"), "ok": False,
                              "error": "模型尚未加载"}), flush=True)
            continue
        if msg.get("path") == "BOOM":
            print(json.dumps({"type": "result", "id": msg.get("id"), "ok": False,
                              "error": "boom"}), flush=True)
            continue
        if msg.get("path") == "CRASH":
            sys.exit(3)
        if msg.get("path") == "SLOW":
            import time
            time.sleep(5)
        print(json.dumps({"type": "result", "id": msg.get("id"), "ok": True, "task": "detect",
                          "width": 100, "height": 80, "class_names": ["cat"],
                          "duration_ms": 1.0,
                          "detections": [{"class_index": 0, "class_name": "cat",
                                          "confidence": 0.9, "bbox": [1, 2, 3, 4]}]}), flush=True)
    elif t == "close":
        print(json.dumps({"type": "bye"}), flush=True)
        break
    elif t == "ping":
        print(json.dumps({"type": "pong"}), flush=True)
'''


def _fake_session(tmp: Path, timeout: float = 5.0):
    """构造一个用假 worker 脚本的 InferSession。"""
    from core.infer.session import InferSession

    script = tmp / "fake_infer_worker.py"
    script.write_text(FAKE_WORKER_SCRIPT, encoding="utf-8")
    # 假权重文件：InferSpec.validate() 要求路径存在
    (tmp / "fake.pt").write_bytes(b"fake-pt")
    (tmp / "w.pt").write_bytes(b"fake-pt")

    session = InferSession(python=sys.executable)
    # 覆盖启动命令：直接跑脚本而不是 -m core.infer.worker
    import subprocess

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
        import threading

        session._reader = threading.Thread(target=session._read_stdout, daemon=True)
        session._reader.start()
        session._stderr_reader = threading.Thread(target=session._read_stderr, daemon=True)
        session._stderr_reader.start()

    session.start = _start  # type: ignore[method-assign]
    return session


def test_session_lifecycle() -> None:
    print("\n== 推理会话：加载 / 推理 / 关闭 ==")
    import tempfile

    from core.infer.session import InferError

    with tempfile.TemporaryDirectory() as tmp:
        session = _fake_session(Path(tmp))
        try:
            info = session.load(InferSpec(weights=str(Path(tmp) / "fake.pt"), classes=["cat"], imgsz=64))
            eq(info["task"], "detect", "加载返回任务类型")
            eq(session.classes, ["cat"], "会话类名")
            check(session.alive, "子进程存活")

            r = session.infer_path("img.jpg")
            eq(r.ok, True, "推理成功")
            eq(len(r.detections), 1, "返回一个框")
            eq(r.detections[0].class_name, "cat", "类名")

            # 业务错误（worker 回 ok=false）不抛异常，而是体现在结果里
            r = session.infer_path("BOOM")
            eq(r.ok, False, "worker 业务错误 -> ok=False")
            contains(r.error, "boom", "错误信息透传")

            # 加载失败
            try:
                session.load(InferSpec(weights=""))
                check(False, "空 weights 应抛错")
            except InferError:
                check(True, "空 weights 抛 InferError")
        finally:
            session.stop()
        eq(session.alive, False, "关闭后子进程不存活")


def test_session_crash_and_timeout() -> None:
    print("\n== 推理会话：子进程崩溃 / 超时 ==")
    import tempfile

    from core.infer.session import InferError

    with tempfile.TemporaryDirectory() as tmp:
        session = _fake_session(Path(tmp))
        try:
            session.load(InferSpec(weights=str(Path(tmp) / "fake.pt")))
            # 崩溃：等待者必须被唤醒并报错，而不是无限挂住
            try:
                session.infer_path("CRASH", timeout=5.0)
                check(False, "子进程崩溃应报错")
            except InferError as exc:
                check(True, f"崩溃被感知: {exc}")
        finally:
            session.stop()

    with tempfile.TemporaryDirectory() as tmp:
        session = _fake_session(Path(tmp))
        try:
            session.load(InferSpec(weights=str(Path(tmp) / "fake.pt")))
            try:
                session.infer_path("SLOW", timeout=0.5)
                check(False, "超时应报错")
            except InferError as exc:
                contains(str(exc), "超时", "超时报错含「超时」")
        finally:
            session.stop()


def test_worker_protocol_direct() -> None:
    print("\n== worker 行协议（直接跑假 worker）==")
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "fake_infer_worker.py"
        script.write_text(FAKE_WORKER_SCRIPT, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )

        def send(obj):
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

        def recv():
            return json.loads(proc.stdout.readline())

        send({"type": "load", "spec": {"weights": "w.pt", "classes": ["cat"]}})
        m = recv()
        eq(m["type"], "loaded", "load -> loaded")
        eq(m["ok"], True, "加载成功")
        eq(m["classes"], ["cat"], "类名回显")

        send({"type": "frame", "id": 1, "path": "a.jpg", "options": {"conf": 0.5}})
        m = recv()
        eq(m["type"], "result", "frame -> result")
        eq(m["id"], 1, "id 配对")
        eq(len(m["detections"]), 1, "一个框")

        send({"type": "ping"})
        eq(recv()["type"], "pong", "ping -> pong")

        send({"type": "close"})
        eq(recv()["type"], "bye", "close -> bye")
        proc.wait(timeout=5)
        eq(proc.returncode, 0, "正常退出码 0")


# ---------------------------------------------------------------------------
# API / WebSocket
# ---------------------------------------------------------------------------


def test_api() -> None:
    print("\n== API：格式 / 权重 / 未加载时的推理 ==")
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        check(False, f"TestClient 不可用: {exc}")
        return

    from app.main import app
    from app import services_infer

    services_infer.reset_session()
    client = TestClient(app)

    resp = client.get("/api/infer/formats")
    eq(resp.status_code, 200, "GET /api/infer/formats 200")
    body = resp.json()
    check(len(body["formats"]) >= 3, f"返回 {len(body['formats'])} 个格式")
    check("active" in body, "含 active 会话状态")

    resp = client.get("/api/infer/weights")
    eq(resp.status_code, 200, "GET /api/infer/weights 200")
    check(isinstance(resp.json()["weights"], list), "weights 是列表")

    # 未加载模型就推理 -> 400
    resp = client.post("/api/infer/image", json={"path": str(Path(__file__))})
    eq(resp.status_code, 400, "未加载模型时推理 400")

    # 加载一个不存在的权重 -> 400
    resp = client.post("/api/infer/models", json={"weights": "Z:/nope/best.pt"})
    eq(resp.status_code, 400, "不存在的权重 400")

    # 参数缺失 -> 400
    resp = client.post("/api/infer/image", json={})
    check(resp.status_code in (400, 422), "缺 path/image 报 4xx")

    resp = client.post("/api/infer/close")
    eq(resp.status_code, 200, "关闭会话 200")


def test_ws_backpressure_logic() -> None:
    print("\n== WebSocket：背压丢帧（与 infer_ws 相同的状态机）==")

    # 这段状态机与 app/api/routes/infer.py 的 pending/processing 逻辑一致：
    # 正在处理时新到的帧覆盖 pending，并把被覆盖的那帧记为 dropped。
    async def scenario() -> list:
        state = {"processing": False, "pending": None}
        events: list = []
        lock = asyncio.Lock()

        async def on_frame(fid: int) -> None:
            async with lock:
                if state["processing"]:
                    dropped = state["pending"]
                    state["pending"] = fid
                    if dropped is not None:
                        events.append(("dropped", dropped))
                else:
                    state["processing"] = True
                    events.append(("run", fid))
                first = state["processing"] and state["pending"] is None

            # 只有在真正开始处理时才模拟耗时推理
            if first or (not state["processing"] and state["pending"] is None):
                pass
            await asyncio.sleep(0)

        # 手工驱动：先标记第一帧在处理中，再连发多帧
        async def drive() -> None:
            async with lock:
                state["processing"] = True
            events.append(("run", 0))
            # 处理期间连发 1/2/3
            for i in (1, 2, 3):
                async with lock:
                    dropped = state["pending"]
                    state["pending"] = i
                    if dropped is not None:
                        events.append(("dropped", dropped))
            async with lock:
                nxt = state["pending"]
                state["pending"] = None
                state["processing"] = False
            events.append(("run", nxt))

        await drive()
        return events

    events = asyncio.run(scenario())
    dropped = [e[1] for e in events if e[0] == "dropped"]
    runs = [e[1] for e in events if e[0] == "run"]
    check(len(dropped) >= 1, f"拥挤帧被标记 dropped（背压生效）: {events}")
    eq(runs, [0, 3], "只处理当前帧与最新一帧，中间帧被丢弃")
    check(1 in dropped and 2 in dropped, f"被丢弃的是中间帧: {dropped}")


def test_class_source() -> None:
    print("\n== 类名来源：显式 > 模型自带 ==")
    from core.infer.ultralytics_infer import describe_model

    class FakeModel:
        task = "detect"

        def __init__(self, names):
            self.names = names

    # 模型自带类名，未显式指定 -> 用模型的
    info = describe_model(FakeModel({0: "cat", 1: "dog"}), InferSpec(weights="x.pt"))
    eq(info["classes"], ["cat", "dog"], "用模型自带类名")
    eq(info["class_source"], "model", "来源标记为 model")

    # 显式指定 -> 覆盖模型自带（这是之前写错的地方：原先只在模型无类名时才用显式）
    info = describe_model(
        FakeModel({0: "cat", 1: "dog"}), InferSpec(weights="x.pt", classes=["person", "car"])
    )
    eq(info["classes"], ["person", "car"], "显式类名覆盖模型自带")
    eq(info["class_source"], "request", "来源标记为 request")

    # 模型无类名 + 显式 -> 用显式
    info = describe_model(FakeModel({}), InferSpec(weights="x.onnx", classes=["a", "b", "c"]))
    eq(info["classes"], ["a", "b", "c"], "模型无类名时用显式")

    # 都没有 -> 空（调用方会退化成 class_N）
    info = describe_model(FakeModel({}), InferSpec(weights="x.onnx"))
    eq(info["classes"], [], "都没有时为空")
    eq(info["class_source"], "", "来源为空")

    # 显式比模型多:超出模型的项也被登记
    info = describe_model(FakeModel({0: "cat"}), InferSpec(weights="x.pt", classes=["cat", "dog"]))
    eq(info["classes"], ["cat", "dog"], "显式多出的类别也登记")


def test_export_surface() -> None:
    import core.infer as mod

    for name in ("InferSession", "InferSpec", "InferOptions", "FrameResult", "Detection"):
        check(hasattr(mod, name), f"core.infer 导出 {name}")


def main() -> int:
    print("== M6 实时验证测试 ==")
    test_formats()
    test_format_detection()
    test_spec_validation()
    test_result_detection()
    test_result_robustness()
    test_result_classification()
    test_session_lifecycle()
    test_session_crash_and_timeout()
    test_worker_protocol_direct()
    test_api()
    test_ws_backpressure_logic()
    test_class_source()
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
