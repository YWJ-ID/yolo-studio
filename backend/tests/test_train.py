"""训练模块测试（无需 pytest，直接运行）。

    cd backend
    .\.venv\Scripts\python.exe tests\test_train.py

覆盖 M2-01 ~ M2-07：
    * TrainerBackend 抽象与 ultralytics 实现（命令行、权重解析、train_kwargs）
    * 任务状态机 start/stop/resume + 子进程隔离 + 服务重启接管
    * results.csv 指标解析（含断点续训去重、半行、增量）
    * WebSocket 事件桥接（监控线程 -> asyncio 队列）
    * 资源监控（无 GPU 时的降级结构）
    * 过程图像分组与目录穿越防护

真实训练用一个"假训练进程"（fixtures/fake_trainer.py）代替，
因为 CPU 上跑真 YOLO 训练无法在测试时长内完成；真实训练另外单独验证。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.train import (  # noqa: E402
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    TrainSpec,
    TrainingManager,
    UltralyticsBackend,
    is_active,
    parse_results_csv,
    resolve_device,
)
from core.train.backend import TrainerBackend  # noqa: E402
from core.train.manager import JobStateError  # noqa: E402

FAKE_TRAINER = Path(__file__).resolve().parent / "fixtures" / "fake_trainer.py"

RESULTS_CSV_SAMPLE = """epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0
1,10.0,2.5,1.5,1.0,0.30,0.20,0.25,0.10,2.2,1.4,0.9,0.01
2,20.0,1.8,1.0,0.7,0.50,0.40,0.45,0.22,1.6,0.9,0.6,0.009
2,21.0,1.7,0.9,0.65,0.55,0.45,0.50,0.25,1.5,0.85,0.55,0.009
3,30.0,1.2,0.6,0.4,0.70,0.60,0.68,0.40,1.0,0.5,0.3,0.008
"""

CLASSIFICATION_CSV = (
    "epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss,lr/pg0\n"
    "1,5.0,0.9,0.6,0.9,0.8,0.01\n"
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
# 测试基础设施
# ---------------------------------------------------------------------------


class FakeBackend(TrainerBackend):
    """用本机解释器跑 fixtures/fake_trainer.py 代替 ultralytics。"""

    name = "fake"
    display_name = "假训练后端"

    def __init__(self, env: dict | None = None) -> None:
        self.env = env or {}

    def resolve_weights(self, weights: str, base_dir=None) -> str:
        return weights or "fake.yaml"

    def build_command(self, spec: TrainSpec):
        return [sys.executable, str(FAKE_TRAINER), str(spec.spec_path)]

    def last_weights(self, run_dir):
        p = Path(run_dir) / "weights" / "last.pt"
        return p if p.is_file() else None


def make_workspace(root: Path) -> Path:
    """建一个可用的最小数据集目录，返回 data.yaml 路径。"""
    data_dir = root / "data"
    (data_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    data_yaml = data_dir / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: [obj]\n", encoding="utf-8")
    return data_yaml


def make_manager(root: Path, backend: TrainerBackend, **kwargs) -> TrainingManager:
    env = getattr(backend, "env", None)
    if env:
        kwargs.setdefault("extra_env", env)
    return TrainingManager(
        root / "runs",
        backend=backend,
        python=sys.executable,
        device="cpu",
        poll_interval=0.15,
        weights_dir=str(root / "weights"),
        **kwargs,
    )


def make_spec(data_yaml: Path, name: str, **overrides) -> TrainSpec:
    fields = dict(
        data_yaml=str(data_yaml),
        project=str(data_yaml.parent.parent / "runs"),
        name=name,
        weights="fake.yaml",
        epochs=3,
        imgsz=64,
        batch=2,
        device="cpu",
    )
    fields.update(overrides)
    return TrainSpec(**fields)


def wait_terminal(manager: TrainingManager, job_id: str, timeout: float = 25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if not is_active(job.status):
            return job
        time.sleep(0.1)
    return manager.get(job_id)


# ---------------------------------------------------------------------------
# M2-01 后端抽象
# ---------------------------------------------------------------------------


def test_backend(tmp: Path) -> None:
    print("\n== TrainerBackend 抽象与 ultralytics 实现 ==")
    backend = UltralyticsBackend()
    spec = make_spec(make_workspace(tmp), "job_backend")

    command = backend.build_command(spec)
    eq(command[1:3], ["-m", "core.train.runner"], "命令行通过 runner 启动")
    contains(command[-1], "train_spec.json", "命令行指向参数文件")
    eq(command[0], sys.executable, "未指定 python 时用当前解释器")

    spec.python = "D:/other/python.exe"
    eq(backend.build_command(spec)[0], "D:/other/python.exe", "spec.python 覆盖解释器")
    spec.python = ""

    # 权重解析
    wdir = tmp / "weights"
    wdir.mkdir(exist_ok=True)
    (wdir / "local.pt").write_bytes(b"x")
    eq(backend.resolve_weights("local.pt", base_dir=wdir), str((wdir / "local.pt").resolve()),
       "在权重目录中按文件名找到权重")
    eq(backend.resolve_weights("yolo11n.yaml"), "yolo11n.yaml", "结构文件原样返回")
    real = wdir / "local.pt"
    eq(backend.resolve_weights(str(real)), str(real.resolve()), "已存在的文件路径原样返回")
    try:
        backend.resolve_weights("", base_dir=wdir)
        check(False, "空权重应报错")
    except ValueError as exc:
        contains(str(exc), "未指定权重", "空权重给出明确提示")

    eq(backend.default_weights("detect"), "yolo11n.yaml", "检测默认权重")
    eq(backend.default_weights("classify"), "yolo11-cls.yaml", "分类默认用 cls 结构文件")

    # train_kwargs
    prepared = backend.train_kwargs(spec)
    eq(prepared["weights"], "fake.yaml", "train_kwargs 会解析权重")
    kwargs = prepared["kwargs"]
    eq(kwargs["device"], "cpu", "设备透传")
    eq(kwargs["workers"], 0, "CPU 训练 workers 默认 0")
    eq(kwargs["exist_ok"], True, "允许使用已创建的训练目录，避免被加后缀")
    check("resume" not in kwargs, "非续训不带 resume 参数")

    spec.resume = True
    check(backend.train_kwargs(spec)["kwargs"].get("resume") is True, "续训时 resume=True")

    spec.resume = False
    spec.lr0 = 0.005
    spec.extra = {"cos_lr": True}
    kwargs = backend.train_kwargs(spec)["kwargs"]
    eq(kwargs["lr0"], 0.005, "lr0 透传")
    eq(kwargs["cos_lr"], True, "extra 参数透传")

    # 校验
    bad = TrainSpec(data_yaml=str(tmp / "missing.yaml"), project=str(tmp), name="x", epochs=0)
    problems = bad.validate()
    check(any("data.yaml" in p for p in problems), "缺少 data.yaml 会被检出")
    check(any("epochs" in p for p in problems), "非法 epochs 会被检出")


# ---------------------------------------------------------------------------
# M2-03 指标解析
# ---------------------------------------------------------------------------


def test_metrics(tmp: Path) -> None:
    print("\n== results.csv 指标解析 ==")
    csv_path = tmp / "results.csv"
    csv_path.write_text(RESULTS_CSV_SAMPLE, encoding="utf-8")

    series = parse_results_csv(csv_path)
    eq(series.epochs, 3, "重复 epoch 只保留一次")
    eq(series.last_epoch, 3, "最后一轮 epoch")
    eq(series.columns[0], "epoch", "解析到列名")

    second = series.rows[1]
    eq(second.epoch, 2, "第二轮 epoch")
    eq(second.get("metrics/mAP50(B)"), 0.5, "同 epoch 保留最后一次写入的值")
    eq(second.headline()["train_loss"], 3.25, "检测损失为三个分项之和")

    best = series.best("metrics/mAP50(B)")
    eq(best.epoch, 3, "按 mAP50 取最优轮次")
    eq(series.best_headline()["best_epoch"], 3, "最优轮次汇总")
    eq(series.progress(6), {"epoch": 3, "epochs_done": 3, "epochs_total": 6, "percent": 0.5},
       "进度按已完成轮数计算")

    incremental = parse_results_csv(csv_path, since_epoch=2)
    eq([r.epoch for r in incremental.rows], [3], "增量解析只返回更新的轮次")

    # 容错
    half = tmp / "half.csv"
    half.write_text(RESULTS_CSV_SAMPLE + "4,40.0,1.0,0.5,0.3", encoding="utf-8")
    eq([r.epoch for r in parse_results_csv(half).rows], [1, 2, 3], "半行（正在写入）被丢弃")
    empty = tmp / "empty.csv"
    empty.write_text("", encoding="utf-8")
    eq(parse_results_csv(empty).epochs, 0, "空文件返回空序列而不是报错")
    eq(parse_results_csv(tmp / "nope.csv").epochs, 0, "文件不存在返回空序列")

    cls_csv = tmp / "cls.csv"
    cls_csv.write_text(CLASSIFICATION_CSV, encoding="utf-8")
    hp = parse_results_csv(cls_csv).rows[0].headline()
    eq(hp.get("top1"), 0.6, "分类任务的 top1 指标被识别")
    eq(hp.get("train_loss"), 0.9, "分类任务直接取 train/loss")
    check("mAP50" not in hp, "分类任务不臆造检测指标")


# ---------------------------------------------------------------------------
# M2-02 状态机
# ---------------------------------------------------------------------------


def test_lifecycle(tmp: Path) -> None:
    print("\n== 任务生命周期（start / 成功 / 失败） ==")
    data_yaml = make_workspace(tmp)
    backend = FakeBackend()
    manager = make_manager(tmp, backend)
    events: list = []

    try:
        spec = make_spec(data_yaml, "job_ok", epochs=3)
        spec.project = str(tmp / "runs")
        job = manager.start(spec)
        eq(job.status, STATUS_RUNNING, "start 后进入 running")
        eq(job.epochs_total, 3, "记录总轮数")
        check(job.pid is not None, "记录了子进程 pid")
        check(Path(job.run_dir, "train_spec.json").is_file(), "参数文件已落盘")

        unsubscribe = manager.subscribe(job.id, events.append)
        final = wait_terminal(manager, job.id)

        eq(final.status, STATUS_FINISHED, "正常退出后状态为 finished")
        eq(final.returncode, 0, "退出码 0")
        eq(final.metrics_rows, 3, "解析到 3 轮指标")
        eq(final.current_epoch, 3, "当前轮次为 3")
        eq(final.progress, 1.0, "进度到 100%")
        eq(final.best.get("best_epoch"), 3, "记录了最优轮次")

        eq(len(manager.metrics(job.id).rows), 3, "指标序列可读取")
        logs = manager.logs(job.id)
        text = "\n".join(l["text"] for l in logs["lines"])
        contains(text, "fake trainer start", "捕获到子进程 stdout")
        contains(text, "fake trainer done", "收尾日志也被捕获")
        check(logs["total"] > 5, "日志行数合理")
        seqs = [l["seq"] for l in logs["lines"]]
        eq(seqs, sorted(seqs), "日志序号单调递增")
        eq(len(set(seqs)), len(seqs), "日志序号不重复")

        kinds = {e["type"] for e in events}
        check("status" in kinds, "推送了 status 事件")
        check("metrics" in kinds, "推送了 metrics 事件")
        check("log" in kinds, "推送了 log 事件")
        check("finished" in kinds, "推送了 finished 事件")
        metric_events = [e for e in events if e["type"] == "metrics"]
        eq(sum(len(e["rows"]) for e in metric_events), 3, "metrics 事件按增量推送，无重复")
        check("artifacts" in [e for e in events if e["type"] == "finished"][0],
              "finished 事件带上产物清单")
        unsubscribe()

        # 落盘可恢复
        job_file = Path(job.run_dir) / "job.json"
        data = json.loads(job_file.read_text(encoding="utf-8"))
        eq(data["status"], STATUS_FINISHED, "job.json 记录了最终状态")

        # 失败路径
        manager_fail = make_manager(tmp / "fail", FakeBackend({"FAKE_EXIT": "3"}))
        try:
            spec_fail = make_spec(data_yaml, "job_fail")
            spec_fail.project = str(tmp / "fail" / "runs")
            manager_fail.start(spec_fail)
            failed = wait_terminal(manager_fail, "job_fail")
            eq(failed.status, STATUS_FAILED, "非零退出码标记为 failed")
            eq(failed.returncode, 3, "记录退出码")
            contains(failed.error, "3", "错误信息包含退出码")
        finally:
            manager_fail.shutdown()

        # 启动参数非法
        try:
            bad = make_spec(tmp / "missing.yaml", "job_bad")
            bad.project = str(tmp / "runs")
            manager.start(bad)
            check(False, "data.yaml 缺失时应抛 ValueError")
        except ValueError as exc:
            contains(str(exc), "data.yaml", "data.yaml 缺失时给出明确错误")

        # 目录非空
        try:
            manager.start(make_spec(data_yaml, "job_ok", epochs=1))
            check(False, "训练目录已存在时应拒绝")
        except ValueError as exc:
            contains(str(exc), "非空", "拒绝覆盖非空训练目录")
    finally:
        manager.shutdown()


def test_stop_and_resume(tmp: Path) -> None:
    print("\n== 停止与断点续训 ==")
    data_yaml = make_workspace(tmp)
    manager = make_manager(tmp, FakeBackend({"FAKE_EPOCHS": "200", "FAKE_DELAY": "0.3"}))
    try:
        spec = make_spec(data_yaml, "job_stop")
        spec.project = str(tmp / "runs")
        manager.start(spec)
        time.sleep(1.0)
        job = manager.stop("job_stop")
        eq(job.status, STATUS_STOPPED, "停止后状态为 stopped")
        eq(job.stopped_by_user, True, "记录为用户主动停止")

        # 未运行的任务不能停止
        try:
            manager.stop("job_stop")
            check(False, "已停止的任务再次 stop 应报错")
        except JobStateError:
            check(True, "已结束的任务再 stop 会给出状态错误")

        # 续训需要 last.pt，停止时尚未产出
        try:
            manager.resume("job_stop")
            check(False, "缺少 last.pt 时续训应报错")
        except JobStateError as exc:
            contains(str(exc), "last.pt", "明确提示缺少断点权重")
    finally:
        manager.shutdown()

    # 正常完成后再续训
    manager2 = make_manager(tmp / "resume", FakeBackend({"FAKE_EPOCHS": "2", "FAKE_DELAY": "0.15"}))
    try:
        spec = make_spec(data_yaml, "job_resume", epochs=2)
        spec.project = str(tmp / "resume" / "runs")
        manager2.start(spec)
        first = wait_terminal(manager2, "job_resume")
        eq(first.status, STATUS_FINISHED, "首轮训练完成")
        eq(first.attempts, 1, "记录尝试次数 1")

        resumed = manager2.resume("job_resume")
        eq(resumed.attempts, 2, "续训后尝试次数为 2")
        second = wait_terminal(manager2, "job_resume")
        eq(second.status, STATUS_FINISHED, "续训完成")
        eq(second.metrics_rows, 4, "两段训练的轮次合并（2+2）")
        eq(second.current_epoch, 4, "续训接在最大 epoch 之后")
        eq(len(manager2.metrics("job_resume").rows), 4, "epoch 未重复计数")
    finally:
        manager2.shutdown()


def test_restart_adoption(tmp: Path) -> None:
    print("\n== 服务重启：接管 / 标记中断 ==")
    data_yaml = make_workspace(tmp)
    runs = tmp / "runs"

    # 进程仍在运行：新管理器应接管
    manager1 = make_manager(tmp, FakeBackend({"FAKE_EPOCHS": "200", "FAKE_DELAY": "0.3"}))
    try:
        spec = make_spec(data_yaml, "job_adopt")
        spec.project = str(runs)
        manager1.start(spec)
        time.sleep(1.2)
        pid = manager1.get("job_adopt").pid
        check(pid is not None, "训练进程已启动")
        manager1.shutdown()  # 不结束子进程

        manager2 = make_manager(tmp, FakeBackend())
        try:
            adopted = manager2.get("job_adopt")
            eq(adopted.status, STATUS_RUNNING, "仍是 running（进程活着）")
            contains(adopted.message, str(pid), "消息里说明接管的 pid")
            stopped = manager2.stop("job_adopt")
            eq(stopped.status, STATUS_STOPPED, "接管后可以正常停止")
        finally:
            manager2.shutdown()
    finally:
        manager1.shutdown()

    # 进程已不在：标记为 interrupted
    fake_runs = tmp / "dead" / "runs" / "job_dead"
    fake_runs.mkdir(parents=True)
    (fake_runs / "train_spec.json").write_text(
        json.dumps(make_spec(data_yaml, "job_dead").to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    (fake_runs / "job.json").write_text(
        json.dumps(
            {
                "id": "job_dead",
                "status": STATUS_RUNNING,
                "spec": make_spec(data_yaml, "job_dead").to_dict(),
                "run_dir": str(fake_runs),
                "created_at": "2026-01-01T00:00:00",
                "pid": 999999999,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager3 = make_manager(tmp / "dead", FakeBackend())
    try:
        dead = manager3.get("job_dead")
        eq(dead.status, STATUS_INTERRUPTED, "进程不存在的运行中任务被标记为 interrupted")
        check(dead.pid is None, "清掉失联的 pid")
    finally:
        manager3.shutdown()


# ---------------------------------------------------------------------------
# M2-05 资源监控
# ---------------------------------------------------------------------------


def test_resources(tmp: Path) -> None:
    print("\n== 资源监控（无 GPU 降级） ==")
    from core.train.resources import ResourceMonitor

    monitor = ResourceMonitor()
    try:
        caps = monitor.capabilities()
        check("cpu_monitor" in caps and "gpu_monitor" in caps, "能力探测包含 CPU 与 GPU")
        check(isinstance(caps.get("gpu_error"), str), "GPU 不可用时给出原因字符串")

        sample = monitor.sample(None)
        check("timestamp" in sample, "采样带时间戳")
        cpu = sample["cpu"]
        eq(cpu.get("available"), True, "psutil 可用时 CPU 采样为 available")
        check(isinstance(cpu.get("percent"), (int, float)), "CPU 使用率是数值")
        check(isinstance(cpu.get("count"), int) and cpu["count"] > 0, "CPU 核心数有效")
        check(isinstance(cpu.get("mem_total"), int) and cpu["mem_total"] > 0, "内存总量有效")

        gpu = sample["gpu"]
        if gpu.get("available"):
            check(len(gpu["devices"]) > 0, "GPU 可用时列出设备")
            check("name" in gpu["devices"][0], "设备信息包含名称")
        else:
            eq(gpu.get("devices"), [], "无 GPU 时设备列表为空而不是缺字段")
            check(bool(gpu.get("error")), "无 GPU 时说明原因")

        # 进程级采样
        proc_sample = monitor.sample(os.getpid())
        check("process" in proc_sample["cpu"], "能采样到指定进程")
        eq(proc_sample["cpu"]["process"]["pid"], os.getpid(), "进程 pid 正确")
    finally:
        monitor.close()

    eq(resolve_device("cpu"), "cpu", "显式设备原样返回")
    eq(resolve_device("cuda:1"), "cuda:1", "显式 CUDA 设备原样返回")
    check(resolve_device("auto") in ("cpu", "cuda:0"), "auto 解析为本机可用设备")


# ---------------------------------------------------------------------------
# M2-06 过程图像
# ---------------------------------------------------------------------------


def test_artifacts(tmp: Path) -> None:
    print("\n== 过程图像分组与访问防护 ==")
    data_yaml = make_workspace(tmp)
    manager = make_manager(tmp, FakeBackend({"FAKE_EPOCHS": "1", "FAKE_DELAY": "0.1"}))
    try:
        spec = make_spec(data_yaml, "job_art", epochs=1)
        spec.project = str(tmp / "runs")
        manager.start(spec)
        wait_terminal(manager, "job_art")

        data = manager.artifacts("job_art")
        eq(data["total"], 6, "6 张过程图被列出")
        group_ids = {g["id"] for g in data["groups"]}
        check("val_pred" in group_ids, "识别出验证集预测图")
        check("train_batch" in group_ids, "识别出训练批次图")
        check("matrix" in group_ids, "识别出混淆矩阵")
        names = {i["name"] for i in data["images"]}
        check("val_batch0_pred.jpg" in names, "预测图在列表中")
        check(all("/" not in i["name"] for i in data["images"]), "只返回文件名，不返回磁盘路径")
        eq(len(data["preview"]), 4, "预览挑选：预测 1 + 标注 1 + 训练批次 2")

        # 解析与防护
        resolved = manager.artifact_path("job_art", "val_batch0_pred.jpg")
        check(resolved is not None and resolved.name == "val_batch0_pred.jpg", "能解析出真实文件")
        eq(manager.artifact_path("job_art", "../job.json"), None, "拒绝目录穿越")
        eq(manager.artifact_path("job_art", "job.json"), None, "拒绝非图像文件")
        eq(manager.artifact_path("job_art", "nope.png"), None, "不存在的文件返回 None")
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# M2-04 API 层与 WebSocket 桥接
# ---------------------------------------------------------------------------


def test_api_layer(tmp: Path) -> None:
    print("\n== API 层（路由函数直调） ==")
    from fastapi import HTTPException

    from app.api.routes import train as train_route
    from app.schemas import TrainRequest

    data_yaml = make_workspace(tmp)
    manager = make_manager(tmp, FakeBackend({"FAKE_EPOCHS": "2", "FAKE_DELAY": "0.15"}))
    train_route.set_manager(manager)
    # 本用例只验证训练接口：把自动评估/注册指向临时目录并关闭自动评估，
    # 避免写进真实的 storage/。
    from app import services
    from core.registry import ModelRegistry

    services.set_managers(registry=ModelRegistry(tmp / "models"))
    services.set_auto_eval(False)
    try:
        # 路由注册表：REST 与 WebSocket 都在，且 SPA 兜底不会吞掉 /api/train
        from app.main import app as fastapi_app

        paths = {getattr(r, "path", None) for r in fastapi_app.routes}
        check("/api/train/jobs" in paths, "训练任务集合路由已注册")
        check("/api/train/jobs/{job_id}/stop" in paths, "停止路由已注册")
        check("/api/train/jobs/{job_id}/resume" in paths, "续训路由已注册")
        check("/api/train/jobs/{job_id}/ws" in paths, "WebSocket 路由已注册")

        backends = train_route.list_backends()
        eq(len(backends.backends), 1, "列出后端")
        check("device" in backends.defaults, "返回默认设备等配置")

        response = train_route.create_job(
            TrainRequest(data_yaml=str(data_yaml), name="api_job", epochs=2, device="cpu")
        )
        eq(response.job["status"], STATUS_RUNNING, "通过接口启动训练")

        jobs = train_route.list_jobs()
        eq(len(jobs.jobs), 1, "任务列表包含新任务")

        detail = train_route.get_job("api_job")
        check("metrics" in detail and "resources" in detail, "详情包含指标与资源")

        wait_terminal(manager, "api_job")
        metrics = train_route.job_metrics("api_job", since_epoch=None)
        eq(len(metrics["metrics"]["rows"]), 2, "接口能取到完整指标")
        incremental = train_route.job_metrics("api_job", since_epoch=1)
        eq([r["epoch"] for r in incremental["metrics"]["rows"]], [2], "接口支持增量取指标")

        logs = train_route.job_logs("api_job", offset=0, limit=10)
        eq(len(logs["lines"]), 10, "日志按 limit 截断")
        eq(logs["next_offset"], 10, "返回下一页偏移")
        tail = train_route.job_logs("api_job", offset=logs["next_offset"], limit=1000)
        eq(tail["next_offset"], tail["total"], "翻到底后 next_offset 等于日志总数")
        eq(len(tail["lines"]), tail["total"] - 10, "剩余日志被完整返回")
        check(tail["total"] >= 10, "日志总数包含训练输出")

        resources = train_route.job_resources("api_job")
        check("cpu" in resources["sample"], "资源接口返回 CPU 数据")

        artifacts = train_route.job_artifacts("api_job")
        check(all(i["url"].startswith("/api/train/jobs/api_job/image") for i in artifacts["images"]),
              "产物图像带受限访问地址")

        try:
            train_route.job_image("api_job", name="../job.json")
            check(False, "带路径分隔的 name 应被拒绝")
        except HTTPException as exc:
            eq(exc.status_code, 400, "目录穿越返回 400")

        try:
            train_route.get_job("nope")
            check(False, "不存在的任务应返回 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "不存在的任务返回 404")

        try:
            train_route.stop_job("api_job")
            check(False, "已结束的任务 stop 应返回 409")
        except HTTPException as exc:
            eq(exc.status_code, 409, "对已结束任务 stop 返回 409")
    finally:
        from app import services

        services.set_auto_eval(True)
        services.reset_managers()
        manager.shutdown()


def test_websocket_bridge(tmp: Path) -> None:
    print("\n== WebSocket 事件桥接（监控线程 -> 事件循环） ==")
    from app.api.routes.train import make_event_bridge

    data_yaml = make_workspace(tmp)
    manager = make_manager(tmp, FakeBackend({"FAKE_EPOCHS": "3", "FAKE_DELAY": "0.2"}))

    async def scenario():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        unsubscribe = manager.subscribe("ws_job", make_event_bridge(loop, queue))

        spec = make_spec(data_yaml, "ws_job", epochs=3)
        spec.project = str(tmp / "runs")
        manager.start(spec)

        collected = []
        deadline = time.time() + 20
        try:
            while time.time() < deadline:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if "finished" in {e["type"] for e in collected}:
                        break
                    continue
                collected.append(event)
                if event["type"] == "finished":
                    break
        finally:
            unsubscribe()
        return collected

    try:
        events = asyncio.run(scenario())
        kinds = [e["type"] for e in events]
        check("metrics" in kinds, "跨线程推送了 metrics 事件")
        check("log" in kinds, "跨线程推送了 log 事件")
        check(kinds[-1] == "finished", "最后一个事件是 finished")
        check(all(e.get("job_id") == "ws_job" for e in events), "事件都带 job_id")
        metric_total = sum(len(e["rows"]) for e in events if e["type"] == "metrics")
        eq(metric_total, 3, "metrics 事件按轮次增量推送，合计 3 轮")
    finally:
        manager.shutdown()

    eq(manager.get("ws_job").status, STATUS_FINISHED, "任务在事件循环中正常跑完")


def test_log_persistence(tmp: Path) -> None:
    print("\n== 日志落盘与重启恢复 ==")
    data_yaml = make_workspace(tmp)
    manager = make_manager(tmp, FakeBackend({"FAKE_EPOCHS": "2", "FAKE_DELAY": "0.15"}))
    try:
        spec = make_spec(data_yaml, "job_logs", epochs=2)
        spec.project = str(tmp / "runs")
        manager.start(spec)
        wait_terminal(manager, "job_logs")

        log_file = Path(manager.get("job_logs").run_dir) / "train.log"
        check(log_file.is_file(), "训练日志写入 train.log")
        text = log_file.read_text(encoding="utf-8")
        contains(text, "fake trainer start", "子进程输出落盘")
        contains(text, "[studio]", "调度信息一并落盘")
    finally:
        manager.shutdown()

    # 新管理器相当于服务重启，应有历史日志
    manager2 = make_manager(tmp, FakeBackend())
    try:
        logs = manager2.logs("job_logs")
        check(logs["total"] > 5, "重启后恢复历史日志")
        check(any("fake trainer start" in l["text"] for l in logs["lines"]), "恢复内容完整")

        events: list = []
        unsubscribe = manager2.subscribe("job_logs", events.append)
        time.sleep(0.6)
        unsubscribe()
        check(not any(e["type"] == "log" for e in events), "历史日志不会被当作增量重复推送")

        eq(manager2.get("job_logs").status, STATUS_FINISHED, "重启后仍能识别已完成状态")
    finally:
        manager2.shutdown()


def test_no_heavy_import(tmp: Path) -> None:
    print("\n== API 进程不加载训练重依赖 ==")
    # 调度与命令行组装路径不应引入 ultralytics / torch。
    # 该断言在独立子进程里跑，避免受本测试文件此前的 import 影响。
    code = (
        "import sys;"
        f"sys.path.insert(0, r'{BACKEND_DIR}');"
        "from core.train import TrainingManager, UltralyticsBackend, TrainSpec;"
        "b=UltralyticsBackend();"
        "s=TrainSpec(data_yaml='x.yaml', project='p', name='n', weights='yolo11n.yaml');"
        "b.build_command(s); b.train_kwargs(s);"
        "print('ultralytics' in sys.modules, 'torch' in sys.modules)"
    )
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(BACKEND_DIR)
    )
    check(out.returncode == 0, f"子进程探测成功 ({out.stderr[-200:] if out.returncode else 'ok'})")
    eq(out.stdout.strip(), "False False", "构建命令与训练参数不加载 ultralytics / torch")


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 训练模块测试")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)

        def sub(name: str) -> Path:
            d = tmp / name
            d.mkdir(parents=True, exist_ok=True)
            return d

        test_backend(sub("backend"))
        test_metrics(sub("metrics"))
        test_lifecycle(sub("lifecycle"))
        test_stop_and_resume(sub("stopresume"))
        test_restart_adoption(sub("restart"))
        test_resources(sub("resources"))
        test_artifacts(sub("artifacts"))
        test_api_layer(sub("api"))
        test_websocket_bridge(sub("ws"))
        test_log_persistence(sub("logs"))
        test_no_heavy_import(sub("noimport"))

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
