"""评估与模型库测试（M3-01 / M3-02），无需 pytest。

    cd backend
    .\.venv\Scripts\python.exe tests\test_eval.py

覆盖：
    * EvalSpec 校验与 EvalResult 归一化 / 序列化往返（纯函数）
    * EvalManager 生命周期：成功 / 失败 / 停止 / 服务重启
    * 混淆矩阵、逐类指标、过程图像与受限访问
    * 训练完成自动评估的联动（app.services 组装）

真实评估（ultralytics val）另外单独验证，因为它需要真跑模型。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.eval import (  # noqa: E402
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    EvalManager,
    EvalResult,
    EvalSpec,
    EvalStateError,
    build_result,
)
from core.train import TrainSpec, TrainingManager, is_active as train_active  # noqa: E402
from core.train.backend import TrainerBackend  # noqa: E402

FAKE_EVALUATOR = Path(__file__).resolve().parent / "fixtures" / "fake_evaluator.py"
FAKE_TRAINER = Path(__file__).resolve().parent / "fixtures" / "fake_trainer.py"

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
# 基础设施
# ---------------------------------------------------------------------------


def make_workspace(root: Path, with_test: bool = True) -> Path:
    """建一个最小数据集目录，返回 data.yaml。"""
    data_dir = root / "data"
    (data_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    lines = ["nc: 2", "names: [square, circle]", "train: images/train", "val: images/val"]
    if with_test:
        (data_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        lines.append("test: images/test")
    data_yaml = data_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def make_weights(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-weights")
    return path


def make_eval_manager(root: Path, env: dict | None = None, **kwargs) -> EvalManager:
    extra_env = dict(env or {})

    def builder(spec: EvalSpec):
        return [sys.executable, str(FAKE_EVALUATOR), str(spec.spec_path)]

    return EvalManager(
        root / "evals",
        python=sys.executable,
        device="cpu",
        poll_interval=0.15,
        extra_env=extra_env,
        command_builder=builder,
        **kwargs,
    )


def make_eval_spec(root: Path, data_yaml: Path, weights: Path, name: str, **overrides) -> EvalSpec:
    fields = dict(
        weights=str(weights),
        data_yaml=str(data_yaml),
        project=str(root / "evals"),
        name=name,
        split="val",
        imgsz=64,
        batch=2,
        device="cpu",
    )
    fields.update(overrides)
    return EvalSpec(**fields)


def wait_eval(manager: EvalManager, eval_id: str, timeout: float = 25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(eval_id)
        if job.status not in ("pending", "running", "stopping"):
            return job
        time.sleep(0.1)
    return manager.get(eval_id)


# ---------------------------------------------------------------------------
# M3-01 spec 与结果归一化
# ---------------------------------------------------------------------------


def test_spec_and_result(tmp: Path) -> None:
    print("\n== EvalSpec 校验与结果归一化 ==")
    data_yaml = make_workspace(tmp)
    weights = make_weights(tmp / "w" / "best.pt")

    spec = make_eval_spec(tmp, data_yaml, weights, "e1")
    eq(spec.validate(), [], "合法参数无问题")
    eq(spec.split, "val", "默认 split 为 val")

    bad = EvalSpec(weights=str(tmp / "nope.pt"), data_yaml=str(tmp / "nope.yaml"), project=str(tmp), name="x")
    problems = bad.validate()
    check(any("权重文件不存在" in p for p in problems), "缺少权重会被检出")
    check(any("data.yaml" in p for p in problems), "缺少 data.yaml 会被检出")

    bad_split = make_eval_spec(tmp, data_yaml, weights, "e2", split="dev")
    check(any("split" in p for p in bad_split.validate()), "非法 split 会被检出")

    # 序列化往返
    restored = EvalSpec.from_dict(spec.to_dict())
    eq(restored.to_dict(), spec.to_dict(), "EvalSpec 序列化往返一致")

    # build_result：从原始 payload 归一
    payload = {
        "ok": True,
        "results_dict": {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.4,
            "metrics/mAP50(B)": 0.6,
            "metrics/mAP50-95(B)": 0.3,
            "fitness": 0.35,
        },
        "per_class": [
            {"index": 0, "name": "square", "instances": 3, "precision": 0.5,
             "recall": 0.4, "f1": 0.44, "ap50": 0.6, "ap50_95": 0.3},
            {"index": 1, "name": "circle", "instances": 2, "precision": 0.6,
             "recall": 0.5, "f1": 0.54, "ap50": 0.7, "ap50_95": 0.4},
        ],
        "speed": {"inference": 7.2},
        "confusion_matrix": {"labels": ["square", "circle", "background"],
                             "matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "artifacts": ["confusion_matrix.png", "BoxPR_curve.png"],
    }
    result = build_result(payload, spec, eval_id="e1", duration_sec=1.25)
    eq(result.ok, True, "结果标记为成功")
    eq(result.overall["mAP50"], 0.6, "results_dict 映射到 mAP50")
    eq(result.overall["mAP50-95"], 0.3, "results_dict 映射到 mAP50-95")
    eq(result.overall["f1"], 0.49, "没有 F1 时按逐类 F1 求均值")
    eq(len(result.per_class), 2, "逐类指标数量正确")
    eq(result.per_class[1].name, "circle", "逐类指标保留类名")
    eq(result.confusion_matrix["matrix"][0][0], 2, "混淆矩阵按整数保存")
    eq(result.confusion_matrix["labels"], ["square", "circle", "background"], "background 固定在标签末位")
    check("rows=预测" in result.confusion_matrix["axis"], "方向标注为 行=预测,列=真实")
    eq(result.model_name, "best.pt", "记录模型文件名")
    eq(result.artifacts, ["confusion_matrix.png", "BoxPR_curve.png"], "保留产物清单")

    # 旧结果兼容：background 曾在标签首位，读取时纠正到最后（矩阵不动）
    legacy = build_result(
        {"ok": True, "confusion_matrix": {"labels": ["background", "square", "circle"],
                                          "matrix": [[0, 1, 0], [0, 2, 0], [0, 0, 1]]}},
        spec, eval_id="e5",
    )
    eq(legacy.confusion_matrix["labels"], ["square", "circle", "background"], "旧结果 background 被纠正到末位")
    eq(legacy.confusion_matrix["matrix"][1][1], 2, "纠正标签不改动矩阵数据")

    # 往返
    path = tmp / "eval_result.json"
    result.save(path)
    loaded = EvalResult.load(path)
    eq(loaded.to_dict(), result.to_dict(), "EvalResult 序列化往返一致")
    eq(loaded.headline()["precision"], 0.5, "headline 只挑关键指标")

    # 分类任务：只认 top1/top5，不臆造检测指标
    cls_payload = {"ok": True, "results_dict": {"metrics/accuracy_top1": 0.8, "metrics/accuracy_top5": 0.95}}
    cls_result = build_result(cls_payload, spec, eval_id="e3")
    eq(cls_result.overall.get("top1"), 0.8, "分类任务识别 top1")
    check("mAP50" not in cls_result.overall, "分类任务不臆造检测指标")

    # 非数值/NaN 容错
    weird = build_result({"ok": True, "results_dict": {"metrics/mAP50(B)": "nan"}}, spec, eval_id="e4")
    check("mAP50" not in weird.overall, "NaN 不写入指标")

    test_extract_metrics(tmp)


class _FakeMetric:
    """模仿 ultralytics 的 Metric：字段是 numpy 数组，且对象带 __len__。"""

    def __init__(self, n: int) -> None:
        import numpy as np

        self.ap_class_index = np.array([0, 1])
        self.p = np.array([0.5, 0.6])
        self.r = np.array([0.4, 0.5])
        self.f1 = np.array([0.44, 0.54])
        self.ap50 = np.array([0.6, 0.7])
        self.ap = np.array([0.3, 0.4])
        self._n = n

    def __len__(self) -> int:
        return self._n


class _FakeResults:
    def __init__(self, nc_classes: int) -> None:
        import numpy as np

        self.names = {0: "square", 1: "circle"}
        self.results_dict = {"metrics/mAP50(B)": np.float64(0.6)}
        self.box = _FakeMetric(nc_classes)
        self.nt_per_class = np.array([3, 2])
        self.speed = {"inference": 7.2}
        self.confusion_matrix = _FakeConfusion()
        self.save_dir = None


class _FakeConfusion:
    def __init__(self) -> None:
        import numpy as np

        self.nc = 2
        self.matrix = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 1]])


def test_extract_metrics(tmp: Path) -> None:
    """extract_metrics 必须能吃 numpy 数组。

    这里刻意用真 numpy 数组当输入：之前用 `value or []` 会在数组上抛
    「truth value of an array is ambiguous」，而只测 build_result 是发现不了的。
    """
    from core.eval.ultralytics_eval import extract_metrics

    payload = extract_metrics(_FakeResults(2))
    eq(len(payload["per_class"]), 2, "从 numpy 数组抽出逐类指标")
    eq(payload["per_class"][0]["name"], "square", "类名映射正确")
    eq(payload["per_class"][0]["instances"], 3, "逐类实例数来自 nt_per_class")
    eq(payload["confusion_matrix"]["matrix"][1][1], 2, "混淆矩阵转成 Python 整数")
    eq(payload["confusion_matrix"]["labels"], ["square", "circle", "background"], "混淆矩阵标签 background 在末位")

    # nc=0（Metric 的 __len__ 为 0）也不能被误判为“没有 metric”
    empty = extract_metrics(_FakeResults(0))
    eq(len(empty["per_class"]), 2, "Metric 对象长度为 0 时仍能取出逐类指标")


# ---------------------------------------------------------------------------
# M3-01 生命周期
# ---------------------------------------------------------------------------


def test_lifecycle(tmp: Path) -> None:
    print("\n== 评估生命周期（成功 / 失败 / 停止） ==")
    data_yaml = make_workspace(tmp)
    weights = make_weights(tmp / "w" / "best.pt")
    manager = make_eval_manager(tmp)
    events: list = []
    try:
        spec = make_eval_spec(tmp, data_yaml, weights, "eval_ok", split="test")
        job = manager.start(spec)
        eq(job.status, STATUS_RUNNING, "启动后进入 running")
        check(Path(job.run_dir, "eval_spec.json").is_file(), "评估参数已落盘")

        unsubscribe = manager.subscribe(job.id, events.append)
        final = wait_eval(manager, "eval_ok")
        eq(final.status, STATUS_FINISHED, "评估成功结束")
        eq(final.returncode, 0, "退出码 0")
        eq(final.metrics.get("split"), "test", "结果记录了划分")
        eq(final.metrics.get("num_classes"), 2, "结果记录了类别数")

        result = manager.result("eval_ok")
        check(result is not None, "能读到结构化结果")
        eq(result.overall["mAP50"], 0.35, "mAP50 = base(0.2)+0.15")
        eq(len(result.per_class), 2, "逐类指标完整")
        eq(result.confusion_matrix["labels"][-1], "background", "混淆矩阵 background 在末位")

        logs = manager.logs("eval_ok")
        text = "\n".join(l["text"] for l in logs["lines"])
        contains(text, "fake evaluator start", "捕获到子进程输出")
        contains(text, "fake evaluator done", "收尾日志也被捕获")

        kinds = {e["type"] for e in events}
        check("status" in kinds and "log" in kinds and "finished" in kinds, "事件类型齐全")
        finished = [e for e in events if e["type"] == "finished"][0]
        check(finished.get("result") is not None, "finished 事件带上结果")
        unsubscribe()

        check((Path(final.run_dir) / "eval.log").is_file(), "评估日志落盘")
        job_file = json.loads((Path(final.run_dir) / "job.json").read_text(encoding="utf-8"))
        eq(job_file["status"], STATUS_FINISHED, "job.json 记录终态")

        # 失败路径
        fail_manager = make_eval_manager(tmp / "fail", {"FAKE_EVAL_EXIT": "3"})
        try:
            fail_manager.start(make_eval_spec(tmp / "fail", data_yaml, weights, "eval_fail"))
            failed = wait_eval(fail_manager, "eval_fail")
            eq(failed.status, STATUS_FAILED, "非零退出码标记为 failed")
            contains(failed.error, "fake evaluation failed", "错误信息来自结果文件")
        finally:
            fail_manager.shutdown()

        # 停止路径
        stop_manager = make_eval_manager(tmp / "stop", {"FAKE_EVAL_DELAY": "30"})
        try:
            stop_manager.start(make_eval_spec(tmp / "stop", data_yaml, weights, "eval_stop"))
            time.sleep(0.8)
            stopped = stop_manager.stop("eval_stop")
            eq(stopped.status, STATUS_STOPPED, "停止后状态为 stopped")
            eq(stopped.stopped_by_user, True, "记录为主动停止")
            try:
                stop_manager.stop("eval_stop")
                check(False, "已结束的评估再次 stop 应报错")
            except EvalStateError:
                check(True, "已结束的评估再 stop 会给出状态错误")
        finally:
            stop_manager.shutdown()

        # 参数非法
        try:
            manager.start(make_eval_spec(tmp, tmp / "missing.yaml", weights, "eval_bad"))
            check(False, "data.yaml 缺失时应抛 ValueError")
        except ValueError as exc:
            contains(str(exc), "data.yaml", "缺失 data.yaml 给出明确错误")
    finally:
        manager.shutdown()


def test_restart_and_artifacts(tmp: Path) -> None:
    print("\n== 服务重启恢复与过程图像 ==")
    data_yaml = make_workspace(tmp)
    weights = make_weights(tmp / "w" / "best.pt")

    # 进程仍在运行：新管理器接管（不粗暴杀掉可能已接近完成的评估）
    manager = make_eval_manager(tmp, {"FAKE_EVAL_DELAY": "30"})
    try:
        manager.start(make_eval_spec(tmp, data_yaml, weights, "eval_long"))
        time.sleep(0.8)
        pid = manager.get("eval_long").pid
        check(pid is not None, "评估进程已启动")
        manager.shutdown()  # 不结束子进程
    finally:
        manager.shutdown()

    manager2 = make_eval_manager(tmp)
    try:
        job = manager2.get("eval_long")
        eq(job.status, STATUS_RUNNING, "服务重启后接管仍在运行的评估")
        contains(job.message, str(pid), "消息里说明接管的 pid")
        stopped = manager2.stop("eval_long")
        eq(stopped.status, STATUS_STOPPED, "接管后可以正常停止")
    finally:
        manager2.shutdown()

    # 进程已不在且没有结果：标记为中断
    dead_dir = tmp / "dead" / "evals" / "eval_dead"
    dead_dir.mkdir(parents=True)
    dead_spec = make_eval_spec(tmp / "dead", data_yaml, weights, "eval_dead")
    (dead_dir / "eval_spec.json").write_text(
        json.dumps(dead_spec.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    (dead_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "eval_dead",
                "status": STATUS_RUNNING,
                "spec": dead_spec.to_dict(),
                "run_dir": str(dead_dir),
                "created_at": "2026-01-01T00:00:00",
                "pid": 999999999,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager_dead = make_eval_manager(tmp / "dead")
    try:
        dead = manager_dead.get("eval_dead")
        eq(dead.status, STATUS_INTERRUPTED, "进程不存在的运行中评估标记为中断")
        check(dead.pid is None, "清掉失联的 pid")
    finally:
        manager_dead.shutdown()

    # 产物与访问防护
    manager3 = make_eval_manager(tmp / "art")
    try:
        spec = make_eval_spec(tmp / "art", data_yaml, weights, "eval_art")
        manager3.start(spec)
        wait_eval(manager3, "eval_art")

        data = manager3.artifacts("eval_art")
        eq(data["total"], 6, "6 张评估过程图被列出")
        groups = {g["id"] for g in data["groups"]}
        check("val_pred" in groups, "识别出验证/测试集预测图")
        check("matrix" in groups, "识别出混淆矩阵")
        check("curve" in groups, "识别出 PR / F1 曲线")

        resolved = manager3.artifact_path("eval_art", "confusion_matrix.png")
        check(resolved is not None and resolved.name == "confusion_matrix.png", "能解析出真实文件")
        eq(manager3.artifact_path("eval_art", "../job.json"), None, "拒绝目录穿越")
        eq(manager3.artifact_path("eval_art", "job.json"), None, "拒绝非图像文件")
    finally:
        manager3.shutdown()


# ---------------------------------------------------------------------------
# M3-01 训练完成自动评估（app.services 组装）
# ---------------------------------------------------------------------------


class FakeTrainBackend(TrainerBackend):
    """复用假的训练进程，让联动测试不依赖真训练。"""

    name = "fake"
    display_name = "假训练后端"

    def resolve_weights(self, weights: str, base_dir=None) -> str:
        return weights or "fake.yaml"

    def build_command(self, spec: TrainSpec):
        return [sys.executable, str(FAKE_TRAINER), str(spec.spec_path)]

    def last_weights(self, run_dir):
        p = Path(run_dir) / "weights" / "last.pt"
        return p if p.is_file() else None


def test_auto_eval(tmp: Path) -> None:
    print("\n== 训练完成自动评估 ==")
    from app import services

    data_yaml = make_workspace(tmp, with_test=True)
    eq(services.pick_split(str(data_yaml)), "test", "声明了 test 就优先评估 test")

    no_test = make_workspace(tmp / "notest", with_test=False)
    eq(services.pick_split(str(no_test)), "val", "没有 test 时退化为 val")

    os.environ["YOLO_STUDIO_AUTO_EVAL_SPLIT"] = "val"
    eq(services.pick_split(str(data_yaml)), "val", "环境变量可强制指定划分")
    os.environ.pop("YOLO_STUDIO_AUTO_EVAL_SPLIT")

    # 联动：训练结束 -> 自动评估
    train_manager = TrainingManager(
        tmp / "runs",
        backend=FakeTrainBackend(),
        python=sys.executable,
        device="cpu",
        poll_interval=0.15,
        extra_env={"FAKE_EPOCHS": "1", "FAKE_DELAY": "0.15"},
    )
    eval_manager = make_eval_manager(tmp, {"FAKE_EVAL_SEED": "2"})
    from core.registry import ModelRegistry

    registry = ModelRegistry(tmp / "models")
    services.set_managers(training=train_manager, evaluation=eval_manager, registry=registry)
    try:
        spec = TrainSpec(
            data_yaml=str(data_yaml),
            project=str(tmp / "runs"),
            name="auto_train",
            weights="fake.yaml",
            epochs=1,
            imgsz=64,
            batch=2,
            device="cpu",
        )
        train_manager.start(spec)
        deadline = time.time() + 25
        while time.time() < deadline and train_active(train_manager.get("auto_train").status):
            time.sleep(0.1)
        trained = train_manager.get("auto_train")
        eq(trained.status, STATUS_FINISHED, "训练完成")

        # 等自动评估出现并结束
        deadline = time.time() + 25
        eval_job = None
        while time.time() < deadline:
            jobs = eval_manager.list()
            if jobs and not is_active_eval(jobs[0].status):
                eval_job = jobs[0]
                break
            time.sleep(0.1)

        check(eval_job is not None, "训练完成后自动创建了评估任务")
        if eval_job is not None:
            eq(eval_job.status, STATUS_FINISHED, "自动评估完成")
            eq(eval_job.spec["job_id"], "auto_train", "评估关联到训练任务")
            eq(eval_job.spec["split"], "test", "自动选择了 test 划分")
            eq(eval_job.spec["weights"], str(Path(trained.run_dir) / "weights" / "best.pt"),
               "自动使用 best.pt")
            result = eval_manager.result(eval_job.id)
            check(result is not None and result.overall, "自动评估产出了指标")

        # 训练完成应当自动注册模型（回调在 finished 事件前触发，这里稍等以容忍调度延迟）
        card = registry.get("auto_train")
        check(card is not None, "训练完成后自动注册了模型")
        if card is not None:
            deadline = time.time() + 5
            while time.time() < deadline and len(registry.get("auto_train").evals) < 1:
                time.sleep(0.1)
            check(len(registry.get("auto_train").evals) >= 1, "自动评估结果挂到了模型卡片上")

        # 关掉自动评估后不应再触发
        services.set_auto_eval(False)
        before = len(eval_manager.list())
        spec2 = TrainSpec(
            data_yaml=str(data_yaml), project=str(tmp / "runs"), name="auto_off",
            weights="fake.yaml", epochs=1, imgsz=64, batch=2, device="cpu",
        )
        train_manager.start(spec2)
        deadline = time.time() + 25
        while time.time() < deadline and train_active(train_manager.get("auto_off").status):
            time.sleep(0.1)
        time.sleep(1.0)
        eq(len(eval_manager.list()), before, "关闭自动评估后不再创建评估任务")
    finally:
        services.set_auto_eval(True)
        services.reset_managers()
        train_manager.shutdown()
        eval_manager.shutdown()


def is_active_eval(status: str) -> bool:
    return status in ("pending", "running", "stopping")


# ---------------------------------------------------------------------------
# API 层
# ---------------------------------------------------------------------------


def test_api_layer(tmp: Path) -> None:
    print("\n== 评估 API 层 ==")
    from fastapi import HTTPException

    from app.api.routes import eval as eval_route
    from app.schemas import EvalRequest

    data_yaml = make_workspace(tmp)
    weights = make_weights(tmp / "w" / "best.pt")
    manager = make_eval_manager(tmp)
    eval_route.set_manager(manager)
    try:
        from app.main import app as fastapi_app

        paths = {getattr(r, "path", None) for r in fastapi_app.routes}
        check("/api/eval/jobs" in paths, "评估任务集合路由已注册")
        check("/api/eval/jobs/{eval_id}/result" in paths, "结果路由已注册")
        check("/api/eval/jobs/{eval_id}/ws" in paths, "评估 WebSocket 路由已注册")

        eq(eval_route.list_splits()["splits"], ["train", "val", "test"], "列出可评估划分")

        response = eval_route.create_job(
            EvalRequest(weights=str(weights), data_yaml=str(data_yaml), split="test", tag="api")
        )
        eq(response.job["status"], STATUS_RUNNING, "通过接口启动评估")
        eval_id = response.job["id"]

        jobs = eval_route.list_jobs()
        eq(len(jobs.jobs), 1, "评估列表包含新任务")

        wait_eval(manager, eval_id)
        detail = eval_route.get_job(eval_id)
        check(detail["result"] is not None, "详情包含结果")
        eq(detail["result"]["split"], "test", "结果 split 正确")

        result_resp = eval_route.job_result(eval_id)
        eq(result_resp.result["overall"]["mAP50"], 0.35, "结果接口返回 mAP50")

        logs = eval_route.job_logs(eval_id, offset=0, limit=5)
        eq(len(logs["lines"]), 5, "日志按 limit 截断")

        artifacts = eval_route.job_artifacts(eval_id)
        check(all(i["url"].startswith("/api/eval/jobs/") for i in artifacts["images"]),
              "产物图像带受限访问地址")

        try:
            eval_route.job_image(eval_id, name="../job.json")
            check(False, "带路径分隔的 name 应被拒绝")
        except HTTPException as exc:
            eq(exc.status_code, 400, "目录穿越返回 400")

        try:
            eval_route.get_job("nope")
            check(False, "不存在的评估应返回 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "不存在的评估返回 404")

        try:
            eval_route.create_job(EvalRequest(weights=str(weights), data_yaml=str(data_yaml), split="dev"))
            check(False, "非法 split 应返回 400")
        except HTTPException as exc:
            eq(exc.status_code, 400, "非法 split 返回 400")

        try:
            eval_route.create_job(EvalRequest())
            check(False, "缺少权重与 job_id 应返回 400")
        except HTTPException as exc:
            eq(exc.status_code, 400, "缺少来源信息返回 400")
    finally:
        from app import services

        services.reset_managers()
        manager.shutdown()


def test_report(tmp: Path) -> None:
    print("\n== 单文件 HTML 评估报告（M3-02） ==")
    from core.eval import render_eval_report, write_eval_report

    data_yaml = make_workspace(tmp)
    weights = make_weights(tmp / "w" / "best.pt")
    manager = make_eval_manager(tmp)
    try:
        manager.start(make_eval_spec(tmp, data_yaml, weights, "eval_report"))
        wait_eval(manager, "eval_report")
        result = manager.result("eval_report")
        run_dir = Path(manager.get("eval_report").run_dir)

        html = render_eval_report(result, title="测试报告", images_dir=run_dir)
        contains(html, "测试报告", "报告含标题")
        contains(html, "<!DOCTYPE html>", "输出完整 HTML 文档")
        contains(html, "逐类指标", "含逐类指标章节")
        contains(html, "square", "含类名")
        contains(html, "混淆矩阵", "含混淆矩阵章节")
        contains(html, "background", "混淆矩阵含 background 行/列")
        contains(html, "各类别 AP50-95", "含逐类 AP 条形图")
        contains(html, "<svg", "图表为内联 SVG")
        contains(html, "data:image/png;base64,", "过程图像以 base64 内嵌")
        contains(html, "rows=预测,cols=真实", "注明混淆矩阵方向（行=预测，列=真实）")

        out = write_eval_report(result, tmp / "report" / "eval.html", images_dir=run_dir)
        check(out.is_file() and out.stat().st_size > 1000, "报告写出且非空")
        plain = render_eval_report(result, embed_images=False, images_dir=run_dir)
        check("data:image/png;base64," not in plain, "可关闭图像内嵌以减小体积")

        # 无结果时不应生成
        from fastapi import HTTPException

        from app.api.routes import eval as eval_route
        from app.schemas import EvalReportRequest

        eval_route.set_manager(manager)
        try:
            response = eval_route.job_report("eval_report", EvalReportRequest(out_name="api_eval_report"))
            check(Path(response.path).is_file(), "通过接口生成报告")
            contains(response.url, "/api/files/download", "返回下载地址")
            check(response.size_bytes > 1000, "报告体积合理")
        finally:
            from app import services

            services.reset_managers()
    finally:
        manager.shutdown()


def test_registry(tmp: Path) -> None:
    print("\n== 模型注册（M3-03） ==")
    from core.registry import ModelRegistry
    from core.train import TrainingJob

    # 造一个"已完成的训练任务"，含 dataset_card.json 血缘
    data_yaml = make_workspace(tmp)
    (data_yaml.parent / "dataset_card.json").write_text(
        json.dumps(
            {
                "name": "demo_ds",
                "task": "detection",
                "classes": ["square", "circle"],
                "created_at": "2026-09-20T10:00:00",
                "images_exported": {"train": 8, "val": 4, "test": 2},
                "sources": [{"source_id": "yolo:demo", "format": "yolo", "root": "/demo"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp / "runs" / "job_a"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"best")
    (run_dir / "weights" / "last.pt").write_bytes(b"last")

    job = TrainingJob(
        id="job_a",
        status="finished",
        run_dir=str(run_dir),
        created_at="2026-09-21T10:00:00",
        ended_at="2026-09-21T10:05:00",
        best={"best_epoch": 3, "best": {"mAP50": 0.62, "mAP50-95": 0.41}},
        attempts=1,
        spec={
            "data_yaml": str(data_yaml),
            "task": "detect",
            "epochs": 3,
            "imgsz": 64,
            "batch": 2,
            "device": "cpu",
            "seed": 42,
            "optimizer": "auto",
            "weights": "yolo11n.yaml",
        },
    )

    registry = ModelRegistry(tmp / "models")
    eq(registry.list(), [], "初始模型库为空")

    card = registry.register_from_training(job)
    eq(card.model_id, "job_a", "模型 id 取自训练任务 id")
    eq(card.dataset.get("name"), "demo_ds", "从 dataset_card.json 反查到数据集版本")
    eq(card.classes, ["square", "circle"], "类别来自数据集卡片")
    eq(card.weights.get("best"), str(run_dir / "weights" / "best.pt"), "记录 best.pt 路径")
    eq(card.training.get("epochs"), 3, "记录训练超参")
    eq(card.training.get("best", {}).get("best", {}).get("mAP50"), 0.62, "记录训练最优指标")
    check(Path(card.model_id and registry.card_path("job_a")).is_file(), "model_card.json 已落盘")

    # 幂等 + 更新
    again = registry.register_from_training(job)
    eq(len(registry.list()), 1, "重复注册不会产生第二张卡片")
    eq(again.model_id, "job_a", "重复注册返回同一模型")

    # 查询与摘要
    loaded = registry.get("job_a")
    eq(loaded.summary()["dataset_name"], "demo_ds", "摘要含数据集名")
    eq(loaded.summary()["has_best"], True, "摘要标明有 best 权重")
    eq(loaded.metric("mAP50"), 0.62, "没有评估时回退到训练最优指标")

    # 挂评估结果
    data_yaml2 = data_yaml
    weights = run_dir / "weights" / "best.pt"
    eval_manager = make_eval_manager(tmp / "evals_for_model")
    try:
        eval_manager.start(
            make_eval_spec(tmp / "evals_for_model", data_yaml2, weights, "eval_for_job_a",
                           split="test", job_id="job_a")
        )
        eval_job = wait_eval(eval_manager, "eval_for_job_a")
        result = eval_manager.result("eval_for_job_a")
        registry.attach_eval("job_a", eval_job, result)

        loaded = registry.get("job_a")
        eq(len(loaded.evals), 1, "评估结果挂到卡片上")
        eq(loaded.evals[0]["split"], "test", "记录评估划分")
        eq(loaded.evals[0]["overall"]["mAP50"], 0.35, "记录评估指标")
        eq(loaded.metric("mAP50", split="test"), 0.35, "按划分取评估指标")
        eq(loaded.summary()["eval_splits"], ["test"], "摘要列出已有评估的划分")

        # 同一 eval_id 重复挂载不重复
        registry.attach_eval("job_a", eval_job, result)
        eq(len(registry.get("job_a").evals), 1, "同一评估重复挂载不重复计数")
    finally:
        eval_manager.shutdown()

    # 非法 id 防目录穿越
    try:
        registry.get("../evil")
        check(False, "非法模型 id 应被拒绝")
    except ValueError:
        check(True, "拒绝目录穿越的模型 id")
    eq(registry.get("nope"), None, "不存在的模型返回 None")


def test_compare(tmp: Path) -> None:
    print("\n== 多模型对比（M3-04） ==")
    from core.eval import compare_eval_results, load_result
    from core.registry import ModelRegistry
    from core.train import TrainingJob

    data_yaml = make_workspace(tmp)
    registry = ModelRegistry(tmp / "models")
    eval_manager = make_eval_manager(tmp / "evals")
    try:
        # 两个"模型"，指标不同（用 FAKE_EVAL_SEED 区分）
        for idx, (job_id, seed) in enumerate((("m1", "1"), ("m2", "3"))):
            run_dir = tmp / "runs" / job_id
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"best")
            job = TrainingJob(
                id=job_id,
                status="finished",
                run_dir=str(run_dir),
                created_at=f"2026-09-2{idx}T10:00:00",
                best={"best_epoch": 1, "best": {"mAP50": 0.1 * (idx + 1)}},
                spec={"data_yaml": str(data_yaml), "task": "detect", "epochs": 1, "imgsz": 64},
            )
            registry.register_from_training(job)

            manager = make_eval_manager(tmp / f"evals_{job_id}", {"FAKE_EVAL_SEED": seed})
            try:
                manager.start(
                    make_eval_spec(tmp / f"evals_{job_id}", data_yaml, run_dir / "weights" / "best.pt",
                                   f"eval_{job_id}", split="test", job_id=job_id)
                )
                eval_job = wait_eval(manager, f"eval_{job_id}")
                registry.attach_eval(job_id, eval_job, manager.result(f"eval_{job_id}"))
            finally:
                manager.shutdown()

        cards = registry.list()
        eq(len(cards), 2, "注册了两个模型")

        items = []
        for card in cards:
            entry = card.eval_for_split("test")
            items.append((card.name, load_result(entry["run_dir"])))
        data = compare_eval_results(items)
        eq(len(data["rows"]), 2, "对比表包含两行")
        eq(data["splits"], ["test"], "两行使用同一划分")
        eq(data["metrics"][0], "mAP50-95", "指标按预设顺序排列")
        check("mAP50" in data["metrics"], "包含 mAP50")
        check(all("values" in r for r in data["rows"]), "每行带指标值")
        check(any(r["values"]["mAP50"] is not None for r in data["rows"]), "至少有模型有 mAP50")
        eq(len(data["class_table"]), 2, "逐类对比表按类别展开")
        eq(len(data["notes"]), 0, "同划分、同类别集合时不产生提示")

        # 不同划分必须提示不可比
        mixed = compare_eval_results([("a", items[0][1]), ("b", items[0][1])])
        entries = []
        for card in cards:
            entries.append(load_result(card.eval_for_split("test")["run_dir"]))
        import copy

        other = copy.deepcopy(entries[0])
        other.split = "val"
        mixed = compare_eval_results([("m1", entries[0]), ("m2", other)])
        check(any("不具备可比性" in n for n in mixed["notes"]), "不同划分会明确提示不可比")

        # 空输入
        empty = compare_eval_results([])
        eq(empty["rows"], [], "没有结果时返回空表并给出提示")
        check(bool(empty["notes"]), "空输入带提示")
    finally:
        eval_manager.shutdown()


def test_models_api(tmp: Path) -> None:
    print("\n== 模型库 API ==")
    from fastapi import HTTPException

    from app.api.routes import models as models_route
    from app.schemas import ModelCompareRequest, ModelRegisterRequest
    from core.registry import ModelRegistry
    from core.train import TrainingJob

    from app.api.routes import train as train_route

    data_yaml = make_workspace(tmp)
    run_dir = tmp / "runs" / "api_model"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"best")

    # 训练任务的 job.json 让 TrainingManager 能取到
    job = TrainingJob(
        id="api_model",
        status="finished",
        run_dir=str(run_dir),
        created_at="2026-09-22T10:00:00",
        spec={"data_yaml": str(data_yaml), "task": "detect", "epochs": 1, "imgsz": 64, "batch": 2},
    )
    (run_dir / "job.json").write_text(
        json.dumps(job.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    train_manager = TrainingManager(
        tmp / "runs", backend=FakeTrainBackend(), python=sys.executable, device="cpu"
    )
    registry = ModelRegistry(tmp / "models")
    train_route.set_manager(train_manager)
    models_route.set_registry(registry)
    try:
        response = models_route.register_model(ModelRegisterRequest(job_id="api_model"))
        eq(response.model["model_id"], "api_model", "通过接口注册模型")

        listed = models_route.list_models()
        eq(len(listed.models), 1, "模型列表包含新模型")
        eq(listed.models[0].has_best, True, "列表标明有 best 权重")

        detail = models_route.get_model("api_model")
        eq(detail.model["task"], "detect", "详情含任务类型")

        try:
            models_route.get_model("nope")
            check(False, "不存在的模型应返回 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "不存在的模型返回 404")

        # 没有评估结果时对比会跳过并提示
        compared = models_route.compare_models(ModelCompareRequest(model_ids=["api_model"], split="test"))
        eq(compared["rows"], [], "没有评估结果时不产生对比行")
        check(any("已跳过" in n for n in compared["notes"]), "对比时说明被跳过的原因")

        try:
            models_route.get_model_result("api_model", split="test")
            check(False, "没有结果时取结果应返回 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "没有评估结果返回 404")

        try:
            models_route.register_model(ModelRegisterRequest(job_id="missing_job"))
            check(False, "不存在的训练任务应返回 404")
        except HTTPException as exc:
            eq(exc.status_code, 404, "不存在的训练任务返回 404")
    finally:
        from app import services

        services.reset_managers()
        train_manager.shutdown()


# ---------------------------------------------------------------------------


def test_register_external(tmp: Path) -> None:
    print("\n== 导入外部模型（.pt） ==")
    from fastapi import HTTPException

    from app.api.routes import models as models_route
    from app.schemas import ModelImportRequest, ModelEvalRequest
    from core.registry import ModelRegistry

    from app import services

    data_yaml = make_workspace(tmp)
    ext_dir = tmp / "external"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ext_weights = ext_dir / "my_best.pt"
    ext_weights.write_bytes(b"fake-pt")

    registry = ModelRegistry(tmp / "models")
    models_route.set_registry(registry)
    try:
        card = registry.register_external(
            weights=str(ext_weights),
            data_yaml=str(data_yaml),
            name="我的外部模型",
            task="detect",
            imgsz=320,
            batch=4,
        )
        check(card.model_id.startswith("ext_"), f"自动生成 ext_ 前缀 id: {card.model_id}")
        eq(card.name, "我的外部模型", "使用展示名")
        eq(card.weights.get("best"), str(ext_weights.resolve()), "记录外部权重路径")
        eq(card.training.get("source"), "external", "标记来源为外部")
        eq(card.training.get("run_dir"), "", "外部模型没有训练目录")
        eq(card.classes, ["square", "circle"], "类别取自 data.yaml 的 names")
        eq(card.dataset.get("data_yaml"), str(data_yaml.resolve()), "记录 data.yaml 绝对路径")
        check(any("外部" in n for n in card.notes), "备注说明是外部导入")

        summary = card.summary()
        eq(summary["source"], "external", "摘要标记来源为外部")
        eq(summary["has_data_yaml"], True, "摘要标明有 data.yaml")
        eq(summary["job_id"], "", "外部模型没有训练任务 id")

        # 同一权重文件重复导入 → 返回同一张卡片，不产生第二张
        again = registry.register_external(weights=str(ext_weights))
        eq(again.model_id, card.model_id, "同一权重重复导入返回同一卡片")
        eq(len(registry.list()), 1, "不产生第二张卡片")

        # 只支持 .pt
        onnx = ext_dir / "m.onnx"
        onnx.write_bytes(b"x")
        try:
            registry.register_external(weights=str(onnx))
            check(False, "非 .pt 应被拒绝")
        except ValueError as exc:
            contains(str(exc), ".pt", "非 .pt 报错说明原因")

        # 权重不存在
        try:
            registry.register_external(weights=str(ext_dir / "nope.pt"))
            check(False, "权重不存在应报错")
        except ValueError as exc:
            contains(str(exc), "不存在", "权重不存在报错")

        # data.yaml 不存在
        try:
            registry.register_external(weights=str(ext_weights), data_yaml=str(tmp / "nope.yaml"))
            check(False, "data.yaml 不存在应报错")
        except ValueError as exc:
            contains(str(exc), "data.yaml", "data.yaml 不存在报错")

        # 非法任务类型
        w2 = ext_dir / "other.pt"
        w2.write_bytes(b"y")
        try:
            registry.register_external(weights=str(w2), task="bogus")
            check(False, "非法任务类型应报错")
        except ValueError as exc:
            contains(str(exc), "任务类型", "非法任务类型报错")

        # 显式 id + 非法字符被清洗 + 冲突
        w3 = ext_dir / "third.pt"
        w3.write_bytes(b"z")
        explicit = registry.register_external(weights=str(w3), model_id="my external!!")
        eq(explicit.model_id, "my_external", "显式 id 里的非法字符被清洗")
        w4 = ext_dir / "fourth.pt"
        w4.write_bytes(b"w")
        try:
            registry.register_external(weights=str(w4), model_id="my_external")
            check(False, "重复的显式 id 应报错")
        except ValueError as exc:
            contains(str(exc), "已存在", "显式 id 冲突报错")

        # 外部模型可以评估（走 /api/models/{id}/eval，依赖卡片里的 data.yaml）
        eval_manager = make_eval_manager(tmp / "evals_ext")
        services.set_managers(evaluation=eval_manager)
        try:
            resp = models_route.eval_model(card.model_id, ModelEvalRequest(split="test"))
            job_id = resp["job"]["id"]
            eval_job = wait_eval(eval_manager, job_id)
            registry.attach_eval(card.model_id, eval_job, eval_manager.result(job_id))
            loaded = registry.get(card.model_id)
            eq(len(loaded.evals), 1, "外部模型的评估结果挂到卡片上")
            eq(loaded.evals[0]["split"], "test", "记录评估划分")
            check("mAP50" in (loaded.evals[0]["overall"] or {}), "记录评估指标")
        finally:
            eval_manager.shutdown()
            services.reset_managers()
            # reset 会把 registry 单例也清掉，这里重新挂回隔离的 registry，
            # 否则后面的 API 调用会落到真实 storage
            models_route.set_registry(registry)

        # 没有 data.yaml 的外部模型：可导入，但评估被拒
        no_data = registry.register_external(weights=str(generate_plain_pt(ext_dir, "nodata")))
        eq(no_data.summary()["has_data_yaml"], False, "未提供 data.yaml 时摘要标明无")
        try:
            models_route.eval_model(no_data.model_id, ModelEvalRequest(split="test"))
            check(False, "没有 data.yaml 时评估应被拒")
        except HTTPException as exc:
            eq(exc.status_code, 400, "没有 data.yaml 评估返回 400")

        # API：正常导入
        wapi = ext_dir / "api_ext.pt"
        wapi.write_bytes(b"api")
        response = models_route.import_model(
            ModelImportRequest(weights=str(wapi), data_yaml=str(data_yaml), name="api 外部")
        )
        check(response.model["model_id"].startswith("ext_"), "通过 API 导入外部模型")
        eq(response.model["training"]["source"], "external", "API 导入标记来源")

        # API：非 .pt → 400
        try:
            models_route.import_model(ModelImportRequest(weights=str(onnx)))
            check(False, "API 导入非 .pt 应返回 400")
        except HTTPException as exc:
            eq(exc.status_code, 400, "API 导入非 .pt 返回 400")
    finally:
        services.reset_managers()


def generate_plain_pt(root: Path, stem: str) -> Path:
    path = root / f"{stem}.pt"
    path.write_bytes(b"fake")
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    print("YOLO Studio 评估模块测试")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)

        def sub(name: str) -> Path:
            d = tmp / name
            d.mkdir(parents=True, exist_ok=True)
            return d

        test_spec_and_result(sub("spec"))
        test_lifecycle(sub("lifecycle"))
        test_restart_and_artifacts(sub("restart"))
        test_auto_eval(sub("auto"))
        test_api_layer(sub("api"))
        test_report(sub("report"))
        test_registry(sub("registry"))
        test_compare(sub("compare"))
        test_models_api(sub("modelsapi"))
        test_register_external(sub("external"))

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
