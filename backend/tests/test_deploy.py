"""部署导出测试（M4-01 / M4-02 / M4-03），无需 pytest。

    cd backend
    .\.venv\Scripts\python.exe tests\test_deploy.py

覆盖：
    * 格式清单与可用性探测（缺包时的原因）
    * DeploySpec 校验、export_kwargs 组装（纯函数）
    * 产物大小 / sha256 记录与完整性校验（M4-03）
    * DeployManager 生命周期：成功 / 部分成功 / 全部失败 / 停止 / 重启接管
    * API 层与产物下载
    * 导出结果挂到模型卡片（与 M3 模型库联动）

真实导出（torchscript）另外单独验证。
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.deploy import (  # noqa: E402
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    DeployManager,
    DeployResult,
    DeploySpec,
    DeployStateError,
    capability_report,
    get_format,
    is_known,
    sha256_file,
    verify_result,
)
from core.deploy.ultralytics_export import export_kwargs  # noqa: E402

FAKE_EXPORTER = Path(__file__).resolve().parent / "fixtures" / "fake_exporter.py"

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


def make_weights(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-pt-weights")
    return path


def make_deploy_manager(
    root: Path, env: dict | None = None, checker=None
) -> DeployManager:
    def builder(spec: DeploySpec):
        return [sys.executable, str(FAKE_EXPORTER), str(spec.spec_path)]

    # 假后端不依赖真实 onnx/openvino，因此只把 rknn 当作不可用，用于验证拒绝路径
    if checker is None:
        checker = lambda fmt: "缺少 Python 包: rknn" if fmt == "rknn" else ""  # noqa: E731

    return DeployManager(
        root / "deploys",
        python=sys.executable,
        device="cpu",
        poll_interval=0.15,
        extra_env=dict(env or {}),
        command_builder=builder,
        format_checker=checker,
    )


def make_spec(root: Path, weights: Path, name: str, **overrides) -> DeploySpec:
    fields = dict(
        weights=str(weights),
        project=str(root / "deploys"),
        name=name,
        formats=["onnx", "torchscript"],
        imgsz=64,
        batch=1,
        device="cpu",
        copy_artifacts=True,
    )
    fields.update(overrides)
    return DeploySpec(**fields)


def wait_deploy(manager: DeployManager, deploy_id: str, timeout: float = 25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(deploy_id)
        if job.status not in ("pending", "running", "stopping"):
            return job
        time.sleep(0.1)
    return manager.get(deploy_id)


# ---------------------------------------------------------------------------


def test_formats() -> None:
    print("\n== 格式清单与可用性探测 ==")
    report = capability_report()
    names = {r["name"] for r in report}
    check({"torchscript", "onnx", "openvino", "engine", "tflite", "coreml", "rknn"} <= names,
          "覆盖主要导出格式")
    check(all("available" in r and "reason" in r for r in report), "每项都带可用性与原因")

    by_name = {r["name"]: r for r in report}
    eq(by_name["torchscript"]["available"], True, "torch 已安装，torchscript 可用")
    eq(by_name["torchscript"]["reason"], "", "可用格式不带失败原因")
    # 不写死某个包是否存在：只校验「可用 <=> 无原因」这条不变式
    for item in report:
        if item["available"]:
            eq(item["reason"], "", f"{item['name']} 可用时不应有原因")
        else:
            check(bool(item["reason"]), f"{item['name']} 不可用时必须说明原因")
    check(any(not i["available"] for i in report), "本机存在不可用的格式（说明探测确实生效）")
    eq(by_name["rknn"]["milestone"], "M4-02", "rknn 归到 M4-02")
    eq(by_name["engine"]["needs_gpu"], True, "engine 标记为需要 GPU")
    eq(by_name["engine"]["available"], False, "无 CUDA / 无 tensorrt 时 engine 不可用")

    eq(is_known("onnx"), True, "onnx 是已知格式")
    eq(is_known("nope"), False, "未知格式被识别")
    eq(get_format("torchscript").suffix, ".torchscript", "格式带产物扩展名")
    try:
        get_format("nope")
        check(False, "未知格式应抛 ValueError")
    except ValueError as exc:
        contains(str(exc), "未知导出格式", "未知格式给出明确错误")


def test_spec_and_kwargs(tmp: Path) -> None:
    print("\n== DeploySpec 与 export_kwargs（纯函数） ==")
    weights = make_weights(tmp / "w" / "best.pt")
    spec = make_spec(tmp, weights, "d1")
    eq(spec.validate(), [], "合法参数无问题")
    eq(spec.normalized_formats(), ["onnx", "torchscript"], "格式去重并保持顺序")

    dup = make_spec(tmp, weights, "d2", formats=["onnx", "onnx", "torchscript"])
    eq(dup.normalized_formats(), ["onnx", "torchscript"], "重复格式被去掉")

    bad = DeploySpec(weights=str(tmp / "nope.pt"), project=str(tmp), name="x", formats=[])
    problems = bad.validate()
    check(any("权重文件不存在" in p for p in problems), "缺少权重会被检出")
    check(any("至少要选择一个导出格式" in p for p in problems), "空格式被检出")

    unknown = make_spec(tmp, weights, "d3", formats=["nope"])
    check(any("未知导出格式" in p for p in unknown.validate()), "未知格式被检出")

    conflict = make_spec(tmp, weights, "d4", half=True, int8=True)
    check(any("half 与 int8" in p for p in conflict.validate()), "half 与 int8 互斥会被检出")

    # 序列化往返
    eq(DeploySpec.from_dict(spec.to_dict()).to_dict(), spec.to_dict(), "DeploySpec 往返一致")

    # export_kwargs：按格式给参数，避免 ultralytics 收到无意义参数
    kwargs = export_kwargs("onnx", spec)
    eq(kwargs["format"], "onnx", "onnx 格式名正确")
    eq(kwargs["simplify"], True, "onnx 带 simplify")
    check("opset" in kwargs, "onnx 带 opset")
    ts = export_kwargs("torchscript", spec)
    check("simplify" not in ts, "torchscript 不带 onnx 专属参数")
    check("opset" not in ts, "torchscript 不带 opset")
    engine = export_kwargs("engine", spec)
    check("int8" in engine, "engine 带 int8")
    spec.extra = {"workspace": 4}
    check(export_kwargs("onnx", spec).get("workspace") == 4, "extra 参数透传")


def test_lifecycle(tmp: Path) -> None:
    print("\n== 导出生命周期（成功 / 部分成功 / 全部失败 / 停止） ==")
    weights = make_weights(tmp / "w" / "best.pt")
    manager = make_deploy_manager(tmp)
    events: list = []
    try:
        spec = make_spec(tmp, weights, "deploy_ok")
        job = manager.start(spec)
        eq(job.status, STATUS_RUNNING, "启动后进入 running")
        check(Path(job.run_dir, "deploy_spec.json").is_file(), "参数已落盘")

        unsubscribe = manager.subscribe(job.id, events.append)
        final = wait_deploy(manager, "deploy_ok")
        eq(final.status, STATUS_FINISHED, "全部成功时 finished")
        eq(final.returncode, 0, "退出码 0")
        eq(len(final.artifacts), 2, "记录了 2 个产物")

        result = manager.result("deploy_ok")
        check(result is not None and result.ok, "能读到结构化结果")
        ok_formats = {a.format for a in result.succeeded()}
        eq(ok_formats, {"onnx", "torchscript"}, "两个格式都成功")
        onnx = next(a for a in result.artifacts if a.format == "onnx")
        check(Path(onnx.path).is_file(), "产物已复制到导出目录")
        eq(onnx.size, Path(onnx.path).stat().st_size, "记录的大小与文件一致")
        eq(onnx.sha256, sha256_file(onnx.path), "记录的 sha256 与文件一致")

        logs = manager.logs("deploy_ok")
        text = "\n".join(l["text"] for l in logs["lines"])
        contains(text, "fake exporter start", "捕获到子进程输出")
        contains(text, "fake exporter done", "收尾日志也被捕获")

        kinds = {e["type"] for e in events}
        check({"status", "log", "finished"} <= kinds, "事件类型齐全")
        finished_event = [e for e in events if e["type"] == "finished"][0]
        check(finished_event.get("result") is not None, "finished 事件带上结果")
        unsubscribe()

        check((Path(final.run_dir) / "deploy.log").is_file(), "日志落盘")
    finally:
        manager.shutdown()

    # 部分成功：某个格式失败仍算 finished
    partial = make_deploy_manager(tmp / "partial", {"FAKE_DEPLOY_FAIL": "onnx"})
    try:
        partial.start(make_spec(tmp / "partial", weights, "deploy_partial"))
        job = wait_deploy(partial, "deploy_partial")
        eq(job.status, STATUS_FINISHED, "部分格式失败仍视为完成")
        result = partial.result("deploy_partial")
        eq(len(result.succeeded()), 1, "1 个格式成功")
        eq(len(result.failed()), 1, "1 个格式失败")
        failed = result.failed()[0]
        eq(failed.format, "onnx", "失败的是 onnx")
        contains(failed.error, "fake export failed", "失败原因被记录")
    finally:
        partial.shutdown()

    # 全部失败
    allfail = make_deploy_manager(tmp / "allfail", {"FAKE_DEPLOY_ALLFAIL": "1"})
    try:
        allfail.start(make_spec(tmp / "allfail", weights, "deploy_fail"))
        job = wait_deploy(allfail, "deploy_fail")
        eq(job.status, STATUS_FAILED, "全部失败时 failed")
        eq(job.returncode, 1, "退出码 1")
        contains(job.error, "所有格式导出失败", "错误信息说明全部失败")
    finally:
        allfail.shutdown()

    # 停止
    slow = make_deploy_manager(tmp / "stop", {"FAKE_DEPLOY_DELAY": "30"})
    try:
        slow.start(make_spec(tmp / "stop", weights, "deploy_stop"))
        time.sleep(0.8)
        stopped = slow.stop("deploy_stop")
        eq(stopped.status, STATUS_STOPPED, "停止后 stopped")
        eq(stopped.stopped_by_user, True, "记录为主动停止")
        try:
            slow.stop("deploy_stop")
            check(False, "已结束的导出再次 stop 应报错")
        except DeployStateError:
            check(True, "已结束的导出再 stop 会给出状态错误")
    finally:
        slow.shutdown()

    # 参数非法 / 未知格式
    guard = make_deploy_manager(tmp / "guard")
    try:
        try:
            guard.start(make_spec(tmp / "guard", tmp / "missing.pt", "deploy_bad"))
            check(False, "权重缺失应抛 ValueError")
        except ValueError as exc:
            contains(str(exc), "权重文件不存在", "权重缺失给出明确错误")

        # 本机不可用的格式在启动前就被拒绝（不启动注定失败的子进程）
        try:
            guard.start(make_spec(tmp / "guard", weights, "deploy_rknn", formats=["rknn"]))
            check(False, "不可用格式应被拒绝")
        except ValueError as exc:
            contains(str(exc), "不可用", "不可用格式给出原因")
    finally:
        guard.shutdown()


def test_verify_and_restart(tmp: Path) -> None:
    print("\n== 产物完整性校验（M4-03）与重启接管 ==")
    weights = make_weights(tmp / "w" / "best.pt")
    manager = make_deploy_manager(tmp)
    try:
        manager.start(make_spec(tmp, weights, "deploy_v"))
        wait_deploy(manager, "deploy_v")

        report = manager.verify("deploy_v")
        eq(report.ok, True, "产物未改动时校验通过")
        eq(report.checked, 2, "校验了 2 个产物")
        eq(report.to_dict()["problems"], [], "没有完整性问题")

        # 篡改产物（保持长度）-> 命中 sha256 校验
        result = manager.result("deploy_v")
        target = next(a for a in result.artifacts if a.format == "onnx")
        original = Path(target.path).read_bytes()
        Path(target.path).write_bytes(b"X" * len(original))
        report2 = manager.verify("deploy_v")
        eq(report2.ok, False, "内容被改动后校验失败")
        eq(report2.problems[0]["issue"], "内容已变化（sha256 不一致）", "指出内容变化")
        check(report2.problems[0]["format"] == "onnx", "定位到具体格式")

        # 长度也变了 -> 先命中大小校验（大小不一致时不必再算哈希）
        Path(target.path).write_bytes(original + b"extra")
        report_size = manager.verify("deploy_v")
        eq(report_size.ok, False, "长度变化后校验失败")
        eq(report_size.problems[0]["issue"], "大小不一致", "长度变化先被大小校验发现")

        # 删除产物 -> 校验失败
        Path(target.path).unlink()
        report3 = manager.verify("deploy_v")
        eq(report3.ok, False, "产物被删除后校验失败")
        eq(report3.problems[0]["issue"], "文件不存在", "指出文件不存在")

        # 纯函数：空结果校验通过
        empty = verify_result(DeployResult(ok=True))
        eq(empty.ok, True, "没有产物时校验通过")
        eq(empty.checked, 0, "没有产物时校验数为 0")
    finally:
        manager.shutdown()

    # 服务重启：进程仍在 -> 接管
    long_manager = make_deploy_manager(tmp / "restart", {"FAKE_DEPLOY_DELAY": "30"})
    try:
        long_manager.start(make_spec(tmp / "restart", weights, "deploy_long"))
        time.sleep(0.8)
        pid = long_manager.get("deploy_long").pid
        check(pid is not None, "导出进程已启动")
        long_manager.shutdown()
    finally:
        long_manager.shutdown()

    manager2 = make_deploy_manager(tmp / "restart")
    try:
        job = manager2.get("deploy_long")
        eq(job.status, STATUS_RUNNING, "服务重启后接管仍在运行的导出")
        contains(job.message, str(pid), "消息里说明接管的 pid")
        eq(manager2.stop("deploy_long").status, STATUS_STOPPED, "接管后可以正常停止")
    finally:
        manager2.shutdown()


def test_api_layer(tmp: Path) -> None:
    print("\n== 导出 API 层 ==")
    from fastapi import HTTPException

    from app.api.routes import deploy as deploy_route
    from app.schemas import DeployRequest

    weights = make_weights(tmp / "w" / "best.pt")
    manager = make_deploy_manager(tmp)
    deploy_route.set_manager(manager)
    try:
        from app.main import app as fastapi_app

        paths = {getattr(r, "path", None) for r in fastapi_app.routes}
        check("/api/deploy/jobs" in paths, "导出任务集合路由已注册")
        check("/api/deploy/jobs/{deploy_id}/verify" in paths, "校验路由已注册")
        check("/api/deploy/jobs/{deploy_id}/ws" in paths, "导出 WebSocket 路由已注册")

        formats = deploy_route.list_formats()
        check(len(formats.formats) >= 5, "列出导出格式")
        check("onnx" in formats.defaults, "默认包含 onnx")
        torchscript = next(f for f in formats.formats if f.name == "torchscript")
        eq(torchscript.available, True, "接口报告 torchscript 可用")

        response = deploy_route.create_job(
            DeployRequest(weights=str(weights), formats=["torchscript"], tag="api")
        )
        eq(response.job["status"], STATUS_RUNNING, "通过接口启动导出")
        deploy_id = response.job["id"]

        eq(len(deploy_route.list_jobs().jobs), 1, "导出列表包含新任务")
        wait_deploy(manager, deploy_id)

        detail = deploy_route.get_job(deploy_id)
        check(detail["result"] is not None, "详情包含结果")
        eq(detail["result"]["artifacts"][0]["format"], "torchscript", "产物格式正确")

        verify = deploy_route.verify_job(deploy_id)
        eq(verify.verify["ok"], True, "校验接口返回通过")

        result_resp = deploy_route.job_result(deploy_id)
        eq(result_resp.result["ok"], True, "结果接口返回成功标记")

        logs = deploy_route.job_logs(deploy_id, offset=0, limit=3)
        eq(len(logs["lines"]), 3, "日志按 limit 截断")

        download = deploy_route.download_artifact(deploy_id, fmt="torchscript")
        eq(Path(download.path).name, "best.torchscript", "下载接口指向产物文件")

        try:
            deploy_route.download_artifact(deploy_id, fmt="onnx")
            check(False, "没有该格式产物时应 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "缺失格式产物返回 404")

        try:
            deploy_route.get_job("nope")
            check(False, "不存在的导出应 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "不存在的导出返回 404")

        try:
            deploy_route.create_job(DeployRequest(weights=str(weights), formats=["rknn"]))
            check(False, "不可用格式应返回 400")
        except HTTPException as exc:
            eq(exc.status_code, 400, "不可用格式返回 400")

        try:
            deploy_route.create_job(DeployRequest())
            check(False, "缺少权重来源应返回 400")
        except HTTPException as exc:
            eq(exc.status_code, 400, "缺少来源信息返回 400")
    finally:
        from app import services

        services.reset_managers()
        manager.shutdown()


def test_model_link(tmp: Path) -> None:
    print("\n== 导出产物挂到模型卡片（与 M3 联动） ==")
    from core.registry import ModelRegistry
    from core.train import TrainingJob

    weights = make_weights(tmp / "runs" / "m1" / "weights" / "best.pt")
    registry = ModelRegistry(tmp / "models")
    job = TrainingJob(
        id="m1", status="finished", run_dir=str(tmp / "runs" / "m1"),
        created_at="2026-09-23T10:00:00",
        spec={"data_yaml": str(tmp / "data.yaml"), "task": "detect", "epochs": 1, "imgsz": 64},
    )
    (tmp / "data.yaml").write_text("nc: 1\nnames: [obj]\n", encoding="utf-8")
    registry.register_from_training(job)
    eq(registry.get("m1").summary()["num_deploys"], 0, "初始没有导出记录")

    manager = make_deploy_manager(tmp, {"FAKE_DEPLOY_DELAY": "0.1"})
    try:
        manager.start(make_spec(tmp, weights, "deploy_link", job_id="m1", formats=["onnx"]))
        deploy_job = wait_deploy(manager, "deploy_link")
        result = manager.result("deploy_link")
        registry.attach_deploy("m1", deploy_job, result)

        card = registry.get("m1")
        eq(len(card.deploys), 1, "导出记录挂到卡片上")
        eq(card.deploys[0]["deploy_id"], "deploy_link", "记录导出 id")
        eq(card.deploys[0]["artifacts"][0]["format"], "onnx", "记录产物格式")
        eq(card.summary()["deploy_formats"], ["onnx"], "摘要列出可用导出格式")

        registry.attach_deploy("m1", deploy_job, result)
        eq(len(registry.get("m1").deploys), 1, "同一导出重复挂载不重复计数")

        # 服务层回调也应能挂载
        from app import services

        services.set_managers(registry=registry)
        try:
            services._after_deploy_finish(deploy_job, result)
            eq(len(registry.get("m1").deploys), 1, "服务层回调幂等")
        finally:
            services.reset_managers()
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 部署导出测试")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)

        def sub(name: str) -> Path:
            d = tmp / name
            d.mkdir(parents=True, exist_ok=True)
            return d

        test_formats()
        test_spec_and_kwargs(sub("spec"))
        test_lifecycle(sub("lifecycle"))
        test_verify_and_restart(sub("verify"))
        test_api_layer(sub("api"))
        test_model_link(sub("link"))

    print("\n" + "=" * 60)
    if _failures:
        print(f"失败 {len(_failures)} 项：")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
