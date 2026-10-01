"""YOLO Studio 命令行入口。

证明 core 层独立可用（不依赖 FastAPI）：

    cd backend
    python -m cli.main scan   "D:\\dataset\\archive"
    python -m cli.main export "D:\\dataset\\archive" --out "storage\\datasets\\dms"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from core.export import ExportConfig, ExportError, export_yolo
from core.ingest import available_adapters, detect_format, load_and_merge, load_dataset
from core.ir import DatasetBundle
from core.split import SplitConfig, assign_splits
from core.train import (
    TASK_DETECT,
    TASKS,
    TrainSpec,
    TrainingManager,
    UltralyticsBackend,
    is_active,
    resolve_device,
)


# ---------------------------------------------------------------------------
# 参数收集
# ---------------------------------------------------------------------------


def _adapter_kwargs(args) -> dict:
    """只收集用户真正指定的适配器参数，避免用 None 覆盖默认值。"""
    kwargs: dict = {}
    if getattr(args, "source_id", None):
        kwargs["source_id"] = args.source_id
    if getattr(args, "group_by", None):
        kwargs["group_by"] = args.group_by
    if getattr(args, "level", None) is not None:
        kwargs["level"] = args.level or None
    if getattr(args, "frames_dir", None):
        kwargs["frames_dir"] = args.frames_dir
    if getattr(args, "include_objects", False):
        kwargs["include_objects"] = True
    if getattr(args, "images_dir", None):
        kwargs["images_dir"] = args.images_dir
    if getattr(args, "voc_one_based", False):
        kwargs["voc_one_based"] = True
    return kwargs


def _split_config(args) -> SplitConfig:
    ratios = tuple(float(x) for x in str(args.split_ratio).split(","))
    if len(ratios) != 3:
        raise SystemExit("--split-ratio 必须是 3 个数，如 0.8,0.1,0.1")
    return SplitConfig(
        ratios=ratios,  # type: ignore[arg-type]
        seed=args.seed,
        strategy="random" if args.no_stratify else "stratified",
        respect_groups=not args.no_groups,
        respect_existing=not args.no_respect_split,
    )


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def _cmd_adapters(_args) -> int:
    for a in available_adapters():
        print(f"  {a['name']:<10} {a['display_name']}")
    return 0


def _cmd_detect(args) -> int:
    root = Path(args.path)
    if not root.exists():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 1
    fmt = detect_format(root)
    print(fmt or "无法识别")
    return 0 if fmt else 2


def _load_bundle(args) -> DatasetBundle:
    """加载一个或多个来源并合并。多个路径时按各自自动探测的格式解析再合并。"""
    paths = args.path if isinstance(args.path, list) else [args.path]
    opts = _adapter_kwargs(args)
    specs = [{"path": p, **opts} for p in paths]
    if len(specs) == 1:
        return load_dataset(specs[0]["path"], **{k: v for k, v in specs[0].items() if k != "path"})
    return load_and_merge(specs)


def _cmd_scan(args) -> int:
    paths = args.path if isinstance(args.path, list) else [args.path]
    bundle: DatasetBundle = _load_bundle(args)

    if len(paths) > 1:
        # 多来源时先逐个报告，让用户看清每个来源解析到了什么
        opts = _adapter_kwargs(args)
        for p in paths:
            single = load_dataset(p, **opts)
            print(f"[来源] {p}")
            print(f"       格式={single.format_name} 形态={single.stats()['annotation_kind']} "
                  f"图像={len(single.images)} 标注={len(single.annotations)} "
                  f"类别={len(single.category_names())}")
        print()

    stats = bundle.stats()
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    print(f"格式      : {stats['format']}")
    print(f"根目录    : {stats['root']}")
    print(f"标注形态  : {stats['annotation_kind']}")
    print(f"图像      : {stats['num_images']}")
    print(f"标注      : {stats['num_annotations']}"
          f"  (检测框 {stats['num_bbox_annotations']} / 图像级 {stats['num_image_labels']})")
    print(f"类别      : {stats['num_categories']}")
    for split, count in stats["split_counts"].items():
        print(f"  {split:<8}: {count}")
    if stats["unassigned_images"]:
        print(f"  未划分  : {stats['unassigned_images']}")
    print("类别分布  :")
    for name, count in sorted(stats["count_by_category"].items(), key=lambda x: -x[1]):
        print(f"  {name:<24} {count}")
    if bundle.warnings:
        print(f"警告      : {len(bundle.warnings)} 条（前 5 条）")
        for w in bundle.warnings[:5]:
            print(f"  - {w}")
    return 0


def _cmd_split(args) -> int:
    bundle = _load_bundle(args)
    report = assign_splits(bundle, _split_config(args))

    print(f"划分单元  : {report.units_total}  (其中分组 {report.units_grouped})")
    print(f"图像总数  : {report.images_total}")
    print(f"已有划分  : {report.images_already_split}")
    print(f"本次分配  : {report.images_assigned}")
    for split in ("train", "val", "test"):
        print(f"  {split:<8}: {report.split_images.get(split, 0)}")
    print("类别分布  :")
    for split in ("train", "val", "test"):
        dist = report.class_by_split.get(split, {})
        if dist:
            items = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            print(f"  {split:<8}: {items}")
    for split, missing in report.classes_missing_in_split.items():
        print(f"  [警告] {split} 缺少类别: {', '.join(missing)}")
    for warn in report.warnings:
        if warn not in ("",):
            print(f"  [警告] {warn}")
    return 0


def _cmd_export(args) -> int:
    bundle = _load_bundle(args)

    # 1) 类别规范化：先统一类别空间、过滤掉不想要的标注形态
    taxonomy_cfg = _taxonomy_config(args)
    has_taxonomy = bool(
        taxonomy_cfg.mapping
        or taxonomy_cfg.keep_classes
        or taxonomy_cfg.drop_classes
        or taxonomy_cfg.kind_filter
        or taxonomy_cfg.class_order
        or taxonomy_cfg.sanitize
    )
    if has_taxonomy:
        from core.taxonomy import apply_taxonomy

        tax_report = apply_taxonomy(bundle, taxonomy_cfg)
        print(f"类别规范  : {len(tax_report.classes_before)} -> "
              f"{len(tax_report.classes_after)} 类"
              + (f"，合并 {len(tax_report.merged)} 项" if tax_report.merged else "")
              + (f"，按形态移除 {tax_report.removed_by_kind} 条标注"
                 if tax_report.removed_by_kind else "")
              + (f"，删除 {tax_report.removed_images} 张空标注图"
                 if tax_report.removed_images else ""))
        for warn in tax_report.warnings[:3]:
            print(f"            [提示] {warn}")

    # 2) 清洗：作用在 IR 上，不修改原始数据
    if args.clean:
        from core.clean import CleanConfig, clean

        clean_config = CleanConfig(
            verify_readable=not args.no_clean_verify,
            check_near_duplicates=not args.no_clean_near,
            min_class_instances=args.clean_min_class,
        )
        clean_report = clean(bundle, clean_config, apply=True)
        print(f"清洗      : 发现 {len(clean_report.findings)} 条问题，"
              f"删图 {clean_report.images_removed}，删标注 {clean_report.annotations_removed}"
              f"（{clean_report.duration_sec:.1f}s）")
        for sev, count in clean_report.counts_by_severity.items():
            if count:
                print(f"            {sev}: {count}")
        for action, count in clean_report.actions_applied.items():
            print(f"            已处置 {action}: {count}")
        for warn in clean_report.warnings[:3]:
            print(f"            [提示] {warn}")

    split_report = None
    if not args.no_split:
        cfg = _split_config(args)
        report = assign_splits(bundle, cfg)
        split_report = report.to_dict()
        print(f"划分: train={report.split_images.get('train', 0)} "
              f"val={report.split_images.get('val', 0)} "
              f"test={report.split_images.get('test', 0)}")

    config = ExportConfig(
        task=args.task,
        name_style=args.name_style,
        file_mode=args.file_mode,
        overwrite=args.overwrite,
        clamp_boxes=not args.no_clamp,
    )
    if args.class_order:
        config.class_order = [c.strip() for c in args.class_order.split(",") if c.strip()]

    try:
        result = export_yolo(bundle, args.out, config, split_report=split_report)
    except ExportError as exc:
        print(f"导出失败: {exc}", file=sys.stderr)
        return 1

    print(f"输出目录  : {result.out_dir}")
    print(f"任务类型  : {result.task}")
    print(f"类别      : {', '.join(result.classes)}")
    for split, count in result.images_exported.items():
        print(f"  {split:<8}: {count}")
    print(f"检测框    : {result.boxes_exported}")
    print(f"跳过      : {result.images_skipped}  {result.skipped_by_reason}")
    print(f"已裁剪框  : {result.clamped_boxes}")
    print(f"data.yaml : {result.data_yaml}")
    for warn in result.warnings:
        print(f"  [警告] {warn}")
    return 0


def _cmd_clean(args) -> int:
    from core.clean import CleanConfig, clean, list_rules

    if args.list_rules:
        for r in list_rules():
            print(f"  [{r['category']}] {r['id']:<28} {r['title']}")
        return 0

    if not args.path:
        print("请指定至少一个数据目录（或使用 --list-rules）", file=sys.stderr)
        return 1

    bundle = _load_bundle(args)
    config = CleanConfig(
        verify_readable=not args.no_verify,
        check_near_duplicates=not args.no_near,
        check_exact_duplicates=not args.no_exact,
        min_class_instances=args.min_class,
        limit=args.limit,
        disabled_rules=args.skip or [],
    )
    if args.near_distance is not None:
        config.near_duplicate_distance = args.near_distance

    report = clean(bundle, config, apply=args.apply)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    mode = "已执行处置" if args.apply else "仅检查（dry-run，未修改任何数据）"
    print(f"模式      : {mode}")
    print(f"图像      : {report.images_before} -> {report.images_after}"
          f"  (-{report.images_removed})")
    print(f"标注      : {report.annotations_before} -> {report.annotations_after}"
          f"  (-{report.annotations_removed})")
    print(f"耗时      : {report.duration_sec:.1f}s")
    print(f"问题总数  : {len(report.findings)}")
    print("按严重程度:")
    for sev, count in report.counts_by_severity.items():
        if count:
            print(f"  {sev:<8}: {count}")
    print("按规则:")
    for rule, count in report.counts_by_rule.items():
        print(f"  {rule:<32} {count}")
    if report.actions_applied:
        print("已执行处置:")
        for action, count in report.actions_applied.items():
            print(f"  {action:<20} {count}")
    if report.warnings:
        print("提示:")
        for w in report.warnings:
            print(f"  - {w}")

    if args.show:
        print(f"\n明细（前 {args.show} 条）:")
        for f in report.findings[: args.show]:
            loc = f.image_path or ""
            print(f"  [{f.severity}] {f.rule}: {f.message}")
            if loc:
                print(f"        {loc}")
    return 0


def _parse_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items or None


def _parse_map(items) -> dict:
    """把 ['旧=新', ...] 解析成映射。"""
    mapping: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--map 需要 `原类名=规范名` 形式，收到: {item!r}")
        src, dst = item.split("=", 1)
        src, dst = src.strip(), dst.strip()
        if not src or not dst:
            raise SystemExit(f"--map 两侧都不能为空: {item!r}")
        mapping[src] = dst
    return mapping


def _taxonomy_config(args):
    from core.taxonomy import TaxonomyConfig

    return TaxonomyConfig(
        mapping=_parse_map(getattr(args, "map", None)),
        keep_classes=_parse_list(getattr(args, "keep", None)),
        drop_classes=_parse_list(getattr(args, "drop", None)) or [],
        kind_filter=getattr(args, "kind", None),
        class_order=_parse_list(getattr(args, "order", None)),
        sanitize=getattr(args, "sanitize", False),
    )


def _cmd_taxonomy(args) -> int:
    from core.taxonomy import apply_taxonomy, suggest_merges, validate_class_names

    bundle = _load_bundle(args)

    if args.suggest:
        print(f"类别清单（{len(bundle.category_names())} 个）:")
        counts = bundle.count_by_category()
        for name in bundle.category_names():
            print(f"  {name:<28} {counts.get(name, 0)}")

        suggestions = suggest_merges(bundle)
        if suggestions:
            print("\n疑似同义类别（可用 --map 合并）:")
            for s in suggestions:
                detail = ", ".join(f"{n}={c}" for n, c in s["counts"].items())
                print(f"  {s['names']}  ->  建议规范名: {s['suggested']}   ({detail})")
        else:
            print("\n未发现疑似同义类别。")

        issues = validate_class_names(bundle.category_names())
        if issues:
            print("\n类名问题:")
            for i in issues:
                print(f"  [{i['level']}] {i['name']}: {i['message']}")
        else:
            print("类名全部合法。")
        return 0

    report = apply_taxonomy(bundle, _taxonomy_config(args))

    print(f"类别      : {len(report.classes_before)} -> {len(report.classes_after)}")
    print(f"  规范化前: {', '.join(report.classes_before)}")
    print(f"  规范化后: {', '.join(report.classes_after)}")
    if report.merged:
        print("  合并关系:")
        for src, dst in sorted(report.merged.items()):
            print(f"    {src} -> {dst}")
    if report.removed_by_kind:
        print(f"  按形态过滤掉的标注: {report.removed_by_kind}")
    if report.removed_by_class:
        print(f"  按类别过滤掉的标注: {report.removed_by_class}")
    if report.removed_images:
        print(f"  因此删除的图像    : {report.removed_images}")
    if report.classes_empty:
        print(f"  已无标注的类别    : {report.classes_empty}")
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    for warn in report.warnings:
        print(f"  [警告] {warn}")
    return 0


def _cmd_report(args) -> int:
    from core.analytics import AnalyticsConfig, analyze, write_html_report

    bundle = _load_bundle(args)
    config = AnalyticsConfig(
        sample_size=args.sample,
        min_class_instances=args.min_class,
    )
    report = analyze(bundle, config)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    s = report.summary
    print(f"图像      : {s['num_images']}")
    print(f"标注      : {s['num_annotations']}"
          f"  (检测框 {s['num_bbox_annotations']} / 图像级 {s['num_image_labels']})")
    print(f"标注形态  : {s['annotation_kind']}")
    print(f"类别      : {s['num_classes']}   分组: {s['num_groups']}")
    print(f"子集      : {s['split_counts']}")
    if s["unassigned_images"]:
        print(f"未划分    : {s['unassigned_images']}")

    print("\n类别分布:")
    for c in report.class_distribution:
        bar = "#" * max(1, int(40 * c["count"] / max(1, report.class_distribution[0]["count"])))
        print(f"  {c['name']:<22} {c['count']:>7}  {c['share']:>6.1%}  {bar}")

    imb = report.imbalance
    if imb:
        print(f"\n不均衡    : 最大/最小 = {imb.get('ratio')} "
              f"({imb.get('max_class')} vs {imb.get('min_class')})，"
              f"头部 20% 类别占 {imb.get('top20pct_share', 0):.1%}")

    if s["num_bbox_annotations"]:
        print("\n目标尺寸:")
        for x in report.size_category:
            print(f"  {x['name']:<24} {x['count']}")

    print("\n类别 × 子集:")
    header = "  " + " " * 22 + "".join(f"{sp:>10}" for sp in ("train", "val", "test"))
    print(header)
    for c in report.class_distribution:
        cells = "".join(
            f"{report.class_by_split.get(sp, {}).get(c['name'], 0):>10}"
            for sp in ("train", "val", "test")
        )
        print(f"  {c['name']:<22}{cells}")

    if report.findings:
        print("\n检查结论:")
        for f in report.findings:
            print(f"  [{f['level']:<7}] {f['message']}")
    if report.recommendations:
        print("\n处置建议:")
        for r in report.recommendations:
            print(f"  - {r}")

    if args.out:
        path = write_html_report(
            report, args.out, title=args.title, embed_samples=not args.no_embed
        )
        size_kb = path.stat().st_size / 1024
        print(f"\nHTML 报告 : {path}  ({size_kb:.1f} KB)")
    return 0


def _cmd_export_ir(args) -> int:
    bundle = _load_bundle(args)
    out = Path(args.out)
    bundle.save_json(out)
    print(f"已导出 IR 快照: {out}")
    return 0


def _cmd_train(args) -> int:
    """启动训练并（默认）等待结束。

    core 层独立可用：这个命令不经过 FastAPI，直接调用 TrainingManager。
    """
    runs_dir = Path(args.runs_dir) if args.runs_dir else Path("storage") / "runs"
    runs_dir = runs_dir.expanduser().resolve()
    device = resolve_device(args.device or os.environ.get("YOLO_STUDIO_DEVICE", "auto"))

    spec = TrainSpec(
        data_yaml=args.data,
        project=str(runs_dir),
        name=args.name or "",
        weights=args.weights or "",
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        optimizer=args.optimizer,
    )

    backend = UltralyticsBackend()
    if args.dry_run:
        if not spec.name:
            from core.train import make_job_id

            spec.name = make_job_id()
        if not spec.weights:
            spec.weights = backend.default_weights(spec.task)
        problems = spec.validate()
        if problems:
            for p in problems:
                print(f"[参数错误] {p}", file=sys.stderr)
            return 1
        from core.train import write_spec

        write_spec(spec)
        print(f"训练目录  : {spec.run_dir}")
        print(f"参数文件  : {spec.spec_path}")
        print(f"权重      : {backend.resolve_weights(spec.weights)}")
        print(f"设备      : {spec.device}")
        print(f"命令      : {' '.join(backend.build_command(spec))}")
        return 0

    manager = TrainingManager(
        runs_dir,
        backend=backend,
        python=sys.executable,
        device=device,
        weights_dir=args.weights_dir,
    )
    try:
        job = manager.start(spec)
    except ValueError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        manager.shutdown()
        return 1

    print(f"任务      : {job.id}")
    print(f"训练目录  : {job.run_dir}")
    print(f"设备      : {device}")

    if args.no_wait:
        print("已启动（未等待）。可在前端或接口查看进度。")
        manager.shutdown()
        return 0

    printed_logs = 0
    try:
        while True:
            current = manager.get(job.id)
            logs = manager.logs(job.id, offset=printed_logs, limit=200)
            for line in logs["lines"]:
                print(f"  {line['text']}")
            printed_logs = logs["next_offset"]

            if not is_active(current.status):
                break
            metrics = manager.metrics(job.id).latest()
            if metrics:
                hp = metrics.headline()
                summary = " ".join(f"{k}={v:.4f}" for k, v in hp.items())
                print(f"  [进度] epoch {metrics.epoch}/{current.epochs_total}  {summary}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n收到中断，正在停止训练……")
        try:
            manager.stop(job.id)
        except Exception:
            pass

    final = manager.get(job.id)
    # 把收尾阶段的日志补打出来
    tail = manager.logs(job.id, offset=printed_logs, limit=1000)
    for line in tail["lines"]:
        print(f"  {line['text']}")

    print(f"状态      : {final.status}  ({final.status_label})")
    if final.returncode is not None:
        print(f"退出码    : {final.returncode}")
    if final.best:
        print(f"最优      : epoch={final.best.get('best_epoch')} {final.best.get('best')}")
    if final.error:
        print(f"错误      : {final.error}", file=sys.stderr)
    artifacts = manager.artifacts(job.id)
    if artifacts["total"]:
        print(f"过程图像  : {artifacts['total']} 张")
    manager.shutdown()
    return 0 if final.status == "finished" else 1


def _cmd_eval(args) -> int:
    """对权重做一次评估（core 层独立可用，不经 FastAPI）。"""
    from core.eval import EvalManager, EvalSpec, is_active as eval_active

    evals_dir = (
        Path(args.evals_dir) if args.evals_dir else Path("storage") / "evals"
    ).expanduser().resolve()
    device = resolve_device(args.device or os.environ.get("YOLO_STUDIO_DEVICE", "auto"))

    spec = EvalSpec(
        weights=args.weights,
        data_yaml=args.data,
        project=str(evals_dir),
        name=args.name or "",
        task=args.task,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        tag=args.tag or "",
    )

    manager = EvalManager(evals_dir, python=sys.executable, device=device)
    try:
        job = manager.start(spec)
    except ValueError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        manager.shutdown()
        return 1

    print(f"评估任务  : {job.id}")
    print(f"评估目录  : {job.run_dir}")
    print(f"划分      : {spec.split}")
    print(f"设备      : {device}")

    printed = 0
    try:
        while True:
            current = manager.get(job.id)
            logs = manager.logs(job.id, offset=printed, limit=200)
            for line in logs["lines"]:
                print(f"  {line['text']}")
            printed = logs["next_offset"]
            if not eval_active(current.status):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n收到中断，正在停止评估……")
        try:
            manager.stop(job.id)
        except Exception:
            pass

    final = manager.get(job.id)
    tail = manager.logs(job.id, offset=printed, limit=1000)
    for line in tail["lines"]:
        print(f"  {line['text']}")

    result = manager.result(job.id)
    print(f"状态      : {final.status}  ({final.status_label})")
    if final.error:
        print(f"错误      : {final.error}", file=sys.stderr)

    if result is not None:
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print("总体指标  :")
            for key, value in result.overall.items():
                print(f"  {key:<12} {value:.4f}")
            if result.per_class:
                print("逐类指标  :")
                print(f"  {'类别':<20}{'实例':>6}{'P':>9}{'R':>9}{'AP50':>9}{'AP50-95':>10}")
                for c in result.per_class:
                    print(
                        f"  {c.name:<20}{c.instances:>6}{c.precision:>9.4f}"
                        f"{c.recall:>9.4f}{c.ap50:>9.4f}{c.ap50_95:>10.4f}"
                    )
            if result.confusion_matrix:
                labels = result.confusion_matrix["labels"]
                print(f"混淆矩阵  : {labels}（{result.confusion_matrix['axis']}）")
                for i, row in enumerate(result.confusion_matrix["matrix"]):
                    name = labels[i] if i < len(labels) else str(i)
                    print(f"  {name:<20}" + "".join(f"{v:>8}" for v in row))
            artifacts = manager.artifacts(job.id)
            if artifacts["total"]:
                print(f"过程图像  : {artifacts['total']} 张 -> {final.run_dir}")

    manager.shutdown()
    return 0 if final.status == "finished" else 1


def _cmd_infer(args) -> int:
    """用权重对图片推理（M6 实时验证的命令行版，不经 FastAPI）。"""
    from core.infer import (
        InferOptions,
        InferSession,
        InferSpec,
        capability_report,
        detect_weights_format,
    )

    if args.list_formats:
        for item in capability_report():
            mark = "可用" if item["available"] else f"不可用：{item['reason']}"
            print(f"  {item['name']:<12} {item['label']:<22} {mark}")
        return 0

    if not args.weights:
        print("请指定 --weights（或使用 --list-formats）", file=sys.stderr)
        return 1
    if not args.images:
        print("请至少给一张图片路径", file=sys.stderr)
        return 1

    device = resolve_device(args.device or os.environ.get("YOLO_STUDIO_DEVICE", "auto"))
    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]

    try:
        fmt = detect_weights_format(args.weights)
    except ValueError as exc:
        print(f"权重格式无法识别: {exc}", file=sys.stderr)
        return 1

    spec = InferSpec(
        weights=args.weights,
        task=args.task or "",
        classes=classes,
        device=device,
        imgsz=args.imgsz,
        fmt=fmt,
        source="cli",
    )
    options = InferOptions(conf=args.conf, iou=args.iou, max_det=args.max_det)

    session = InferSession(python=sys.executable)
    payloads = []
    try:
        info = session.load(spec)
        print(f"权重      : {args.weights}")
        print(f"格式      : {fmt}   任务: {info['task']}   设备: {device}")
        print(f"类别({info['num_classes']})  : {', '.join(info['classes']) or '(无)'}")
        print("")

        for img in args.images:
            result = session.infer_path(img, options)
            payloads.append({"image": img, "result": result.to_dict()})

            if result.task == "classify":
                top1 = result.top1 or {}
                print(
                    f"  {Path(img).name:<28} {top1.get('class_name', '?')}"
                    f"  ({top1.get('confidence')})  {result.duration_ms} ms"
                )
            else:
                print(f"  {Path(img).name:<28} {len(result.detections)} 个框  {result.duration_ms} ms")
                for det in result.detections[:10]:
                    x1, y1, x2, y2 = det.bbox
                    print(
                        f"      {det.class_name:<16} conf={det.confidence}"
                        f"  [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
                    )
    finally:
        session.stop()

    if args.save:
        _save_annotated(args.save, payloads)

    if args.json:
        print(json.dumps(payloads, ensure_ascii=False, indent=2))
    return 0


def _save_annotated(out_dir: str, payloads: List[dict]) -> None:
    """把带框的结果图保存下来（CLI 专用的事后可视化）。"""
    try:
        import cv2
    except Exception as exc:
        print(f"保存结果图需要 opencv: {exc}", file=sys.stderr)
        return

    target = Path(out_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for entry in payloads:
        img_path = entry["image"]
        result = entry["result"]
        image = cv2.imread(img_path)
        if image is None:
            continue
        for det in result["detections"]:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{det['class_name']} {det['confidence']}"
            cv2.putText(image, label, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        dst = target / Path(img_path).name
        cv2.imwrite(str(dst), image)
        print(f"  已保存: {dst}")


def _cmd_prelabel(args) -> int:
    """用权重给一个图片目录批量打框，产出**伪标签**（M7，core 层独立可用）。

    产出的是模型预测，不是标注：漏检 = 缺标注、误检 = 错标注，
    **必须人工复核后**才能作为训练数据。给出 --out 时顺手导出为数据集。
    """
    from core.infer import capability_report, detect_weights_format
    from core.export import ExportConfig, ExportError, export_yolo
    from core.prelabel import (
        PrelabelConfig,
        ensure_prelabel_prefix,
        run_prelabel,
    )
    from core.split import SplitConfig, assign_splits

    if args.list_formats:
        for item in capability_report():
            mark = "可用" if item["available"] else f"不可用：{item['reason']}"
            print(f"  {item['name']:<12} {item['label']:<22} {mark}")
        return 0

    if not args.weights:
        print("请指定 --weights（或使用 --list-formats）", file=sys.stderr)
        return 1
    if not args.images:
        print("请指定待标注的图片目录 --images", file=sys.stderr)
        return 1

    if not Path(args.images).exists():
        print(f"图片目录不存在: {args.images}", file=sys.stderr)
        return 1

    device = resolve_device(args.device or os.environ.get("YOLO_STUDIO_DEVICE", "auto"))
    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    try:
        fmt = detect_weights_format(args.weights)
    except ValueError as exc:
        print(f"权重格式无法识别: {exc}", file=sys.stderr)
        return 1

    config = PrelabelConfig(
        weights=args.weights,
        classes=classes,
        device=device,
        imgsz=args.imgsz,
        task=args.task,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        fmt=fmt,
    )

    print("提示：模型给的是**伪标签**，漏检会变成「没有这个标注」，误检会被固化成标注。")
    print("      这批结果必须经人工复核后才能作为训练数据。")
    print("")

    def on_progress(done: int, total: int, current: str) -> None:
        if done == total or done % 10 == 0:
            print(f"  [{done}/{total}] {current}")

    bundle, report = run_prelabel(
        args.images, config, limit=args.limit, on_progress=on_progress
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"权重      : {report.weights}")
        print(f"任务      : {report.task}   设备: {report.device}   conf: {report.conf}")
        print(f"类别({len(report.classes)})  : {', '.join(report.classes) or '(无)'}")
        print(f"图像      : {report.images_total}"
              f"  (有标注 {report.images_with_boxes} / 空 {report.images_empty}"
              f" / 失败 {report.images_failed})")
        print(f"标注      : {report.boxes_total}")
        if report.count_by_category:
            print("逐类实例  :")
            for name, count in sorted(report.count_by_category.items(), key=lambda x: -x[1]):
                print(f"  {name:<24} {count}")
        if report.failures:
            print(f"失败明细  : {len(report.failures)} 张（前 5）")
            for item in report.failures[:5]:
                print(f"  - {item['path']}: {item['error']}")
        print(f"耗时      : {report.duration_sec:.1f}s")

    if not report.ok:
        print(f"错误      : {report.error}", file=sys.stderr)
        return 1

    if not args.out:
        return 0

    # 导出：目录名强制带 prelabel 前缀，其余全部复用现成流水线
    out = Path(args.out).expanduser()
    out = out.parent / ensure_prelabel_prefix(out.name)
    out = out.resolve()

    split_report = None
    if not args.no_split:
        ratios = tuple(float(x) for x in str(args.split_ratio).split(","))
        if len(ratios) != 3:
            print("--split-ratio 必须是 3 个数，如 0.8,0.1,0.1", file=sys.stderr)
            return 1
        split_report = assign_splits(
            bundle,
            SplitConfig(
                ratios=ratios,  # type: ignore[arg-type]
                seed=args.seed,
                strategy="random" if args.no_stratify else "stratified",
                respect_groups=not args.no_groups,
                respect_existing=not args.no_respect_split,
            ),
        ).to_dict()

    try:
        result = export_yolo(bundle, out, ExportConfig(overwrite=args.overwrite), split_report=split_report)
    except ExportError as exc:
        print(f"导出失败: {exc}", file=sys.stderr)
        return 1

    print("")
    print(f"导出目录  : {result.out_dir}")
    print(f"任务类型  : {result.task}")
    print(f"类别      : {', '.join(result.classes)}")
    for split, count in result.images_exported.items():
        print(f"  {split:<8}: {count}")
    print(f"检测框    : {result.boxes_exported}")
    print(f"data.yaml : {result.data_yaml}")
    print(f"血缘      : {result.dataset_card}（含 prelabel 段）")
    print("提醒      : 导出的是伪标签数据集，训练前请人工复核。")
    return 0


def _cmd_export_model(args) -> int:
    """把权重导出为部署格式（core 层独立可用，不经 FastAPI）。"""
    from core.deploy import DeployManager, DeploySpec, capability_report, is_active as deploy_active

    if args.list_formats:
        for item in capability_report():
            mark = "可用" if item["available"] else f"不可用：{item['reason']}"
            print(f"  {item['name']:<12} {item['label']:<18} [{item['milestone']}] {mark}")
        return 0

    if not args.weights:
        print("请指定 --weights（或使用 --list-formats）", file=sys.stderr)
        return 1

    deploys_dir = (
        Path(args.deploys_dir) if args.deploys_dir else Path("storage") / "deploys"
    ).expanduser().resolve()
    device = resolve_device(args.device or os.environ.get("YOLO_STUDIO_DEVICE", "auto"))
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]

    spec = DeploySpec(
        weights=args.weights,
        project=str(deploys_dir),
        name=args.name or "",
        formats=formats,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
        opset=args.opset,
        int8=args.int8,
        nms=args.nms,
        copy_artifacts=not args.no_copy,
        tag=args.tag or "",
    )

    manager = DeployManager(deploys_dir, python=sys.executable, device=device)
    try:
        job = manager.start(spec)
    except ValueError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        manager.shutdown()
        return 1

    print(f"导出任务  : {job.id}")
    print(f"导出目录  : {job.run_dir}")
    print(f"格式      : {', '.join(formats)}")
    print(f"设备      : {device}")

    printed = 0
    try:
        while True:
            current = manager.get(job.id)
            logs = manager.logs(job.id, offset=printed, limit=200)
            for line in logs["lines"]:
                print(f"  {line['text']}")
            printed = logs["next_offset"]
            if not deploy_active(current.status):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n收到中断，正在停止导出……")
        try:
            manager.stop(job.id)
        except Exception:
            pass

    final = manager.get(job.id)
    tail = manager.logs(job.id, offset=printed, limit=1000)
    for line in tail["lines"]:
        print(f"  {line['text']}")

    print(f"状态      : {final.status}  ({final.status_label})")
    if final.error:
        print(f"错误      : {final.error}", file=sys.stderr)

    result = manager.result(job.id)
    if result is not None:
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print("产物      :")
            for a in result.artifacts:
                if a.ok:
                    size_mb = a.size / 1024 / 1024
                    print(f"  [ok]   {a.format:<12} {size_mb:>8.2f} MB  {a.path}")
                else:
                    print(f"  [fail] {a.format:<12} {a.error}")
            report = manager.verify(job.id)
            print(f"完整性    : {'通过' if report.ok else '发现问题'}"
                  f"（校验 {report.checked} 个产物）")
            for problem in report.problems:
                print(f"  [问题] {problem['format']}: {problem['issue']}")

    manager.shutdown()
    return 0 if final.status == "finished" else 1


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def _add_adapter_args(p) -> None:
    p.add_argument("--fmt", default=None, help="强制指定格式")
    p.add_argument("--source-id", default=None)
    p.add_argument("--group-by", default=None,
                   help="分组键：none | parent | stem | regex:<正则>（正则取第一个捕获组）")
    p.add_argument("--level", default=None,
                   help="OpenLABEL: 取哪个标注层级作为类别 (默认 driver_actions；传空串表示全部)")
    p.add_argument("--frames-dir", default=None, help="OpenLABEL: 帧图片目录")
    p.add_argument("--include-objects", action="store_true", help="OpenLABEL: 并入 object 类型标注")
    p.add_argument("--images-dir", default=None,
                   help="COCO/VOC: 图像所在目录（默认自动推断）")
    p.add_argument("--voc-one-based", action="store_true",
                   help="VOC: 坐标按 1-based 闭区间处理（原始 VOC devkit）；默认按 LabelImg 的 0-based")


def _add_taxonomy_args(p) -> None:
    p.add_argument("--map", action="append", default=None,
                   help="类别映射，格式 原类名=规范名，可重复")
    p.add_argument("--keep", default=None, help="逗号分隔，只保留这些类别")
    p.add_argument("--drop", default=None, help="逗号分隔，丢弃这些类别")
    p.add_argument("--kind", default=None, choices=[None, "bbox", "image"],
                   help="只保留该标注形态（bbox=检测框，image=图像级）")
    p.add_argument("--order", default=None, help="逗号分隔，显式指定类别顺序")
    p.add_argument("--sanitize", action="store_true",
                   help="把类名改造为可安全用作目录名的形式")


def _add_split_args(p) -> None:
    p.add_argument("--split-ratio", default="0.8,0.1,0.1", help="train,val,test 比例")
    p.add_argument("--seed", type=int, default=42, help="划分随机种子")
    p.add_argument("--no-stratify", action="store_true", help="关闭分层划分")
    p.add_argument("--no-groups", action="store_true",
                   help="不按 group 分组（危险：视频抽帧数据会产生泄漏）")
    p.add_argument("--no-respect-split", action="store_true", help="忽略输入里已有的划分")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo-studio", description="YOLO Studio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("adapters", help="列出已支持的数据格式")
    p.set_defaults(func=_cmd_adapters)

    p = sub.add_parser("detect", help="探测目录数据格式")
    p.add_argument("path")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("scan", help="扫描数据集并输出统计（可传多个来源，会合并统计）")
    p.add_argument("path", nargs="+", help="一个或多个数据目录")
    _add_adapter_args(p)
    p.add_argument("--json", action="store_true", help="以 JSON 输出")
    p.set_defaults(func=_cmd_scan)

    p = sub.add_parser("split", help="划分数据集并查看分布（不落盘；可传多个来源先合并）")
    p.add_argument("path", nargs="+", help="一个或多个数据目录")
    _add_adapter_args(p)
    _add_split_args(p)
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("export", help="合并多个来源 + 划分 + 导出为 ultralytics 数据集")
    p.add_argument("path", nargs="+", help="一个或多个数据目录，会合并为同一个数据集")
    p.add_argument("--out", required=True, help="输出目录")
    _add_adapter_args(p)
    _add_taxonomy_args(p)
    _add_split_args(p)
    p.add_argument("--no-split", action="store_true", help="不重新划分，沿用输入自带划分")
    p.add_argument("--clean", action="store_true",
                   help="导出前先清洗（执行处置：裁剪越界框、去重、删泄漏样本）")
    p.add_argument("--no-clean-verify", action="store_true", help="清洗时跳过图像解码校验")
    p.add_argument("--no-clean-near", action="store_true", help="清洗时跳过近似重复检测")
    p.add_argument("--clean-min-class", type=int, default=5, help="清洗时的类别最少实例数")
    p.add_argument("--task", default="auto", choices=["auto", "detection", "classification"])
    p.add_argument("--name-style", default="keep", choices=["keep", "source", "uid"])
    p.add_argument("--file-mode", default="copy", choices=["copy", "hardlink", "symlink"],
                   help="copy 最通用；hardlink 最快（需同一磁盘）")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖非空输出目录")
    p.add_argument("--no-clamp", action="store_true", help="不裁剪越界框")
    p.add_argument("--class-order", default=None, help="逗号分隔的类别顺序")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("taxonomy", help="查看/规范化类别体系（合并同义类名、过滤标注形态）")
    p.add_argument("path", nargs="+", help="一个或多个数据目录")
    _add_adapter_args(p)
    _add_taxonomy_args(p)
    p.add_argument("--suggest", action="store_true",
                   help="只查看类别清单与同义类名建议，不做任何改动")
    p.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    p.set_defaults(func=_cmd_taxonomy)

    p = sub.add_parser("clean", help="检查（并可处置）数据质量问题")
    p.add_argument("path", nargs="*", help="一个或多个数据目录")
    _add_adapter_args(p)
    p.add_argument("--apply", action="store_true",
                   help="执行处置（默认仅 dry-run 报告，不修改任何数据）")
    p.add_argument("--list-rules", action="store_true", help="只列出全部清洗规则")
    p.add_argument("--no-verify", action="store_true", help="跳过图像可解码性校验（快很多）")
    p.add_argument("--no-near", action="store_true", help="跳过近似重复检测（pHash 较慢）")
    p.add_argument("--no-exact", action="store_true", help="跳过完全重复检测")
    p.add_argument("--near-distance", type=int, default=None, help="pHash 汉明距离阈值，默认 6")
    p.add_argument("--min-class", type=int, default=5, help="类别最少实例数，默认 5")
    p.add_argument("--limit", type=int, default=0, help="只检查前 N 张（快速预览）")
    p.add_argument("--skip", action="append", default=None, help="关闭某条规则，可重复")
    p.add_argument("--show", type=int, default=0, help="打印前 N 条明细")
    p.add_argument("--json", action="store_true", help="以 JSON 输出")
    p.set_defaults(func=_cmd_clean)

    p = sub.add_parser("report", help="统计分析与质量报告（可导出单文件 HTML）")
    p.add_argument("path", nargs="+", help="一个或多个数据目录")
    _add_adapter_args(p)
    p.add_argument("--out", default=None, help="导出 HTML 报告的路径")
    p.add_argument("--title", default="数据集质量报告", help="报告标题")
    p.add_argument("--sample", type=int, default=24, help="抽样预览张数，默认 24")
    p.add_argument("--min-class", type=int, default=20, help="类别实例数低于此值则提示样本不足")
    p.add_argument("--no-embed", action="store_true", help="HTML 报告不内嵌缩略图（体积更小）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出全部统计结果")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("export-ir", help="把解析结果导出为 IR JSON 快照")
    p.add_argument("path")
    p.add_argument("--out", required=True)
    _add_adapter_args(p)
    p.set_defaults(func=_cmd_export_ir)

    p = sub.add_parser("train", help="启动训练（core 层直接调度，可 --dry-run 只看命令）")
    p.add_argument("--data", required=True, help="数据集 data.yaml 路径")
    p.add_argument("--weights", default="", help="权重：.yaml 从零训练 / .pt 预训练；默认按 task 取")
    p.add_argument("--task", default=TASK_DETECT, choices=list(TASKS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="cpu | cuda:0；默认读 YOLO_STUDIO_DEVICE（auto 时自动探测）")
    p.add_argument("--workers", type=int, default=0, help="dataloader 进程数，CPU 训练建议 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--name", default="", help="任务名/训练目录名，默认自动生成")
    p.add_argument("--runs-dir", default="", help="训练产物根目录，默认 backend/storage/runs")
    p.add_argument("--weights-dir", default="", help="预训练权重目录")
    p.add_argument("--no-wait", action="store_true", help="只启动，不等待训练结束")
    p.add_argument("--dry-run", action="store_true", help="只输出将要执行的命令，不启动训练")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="评估权重（val / test），输出总体与逐类指标")
    p.add_argument("--weights", required=True, help="权重文件路径（如 storage\\runs\\<job>\\weights\\best.pt）")
    p.add_argument("--data", required=True, help="数据集 data.yaml 路径")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--task", default=TASK_DETECT, choices=list(TASKS))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="cpu | cuda:0；默认读 YOLO_STUDIO_DEVICE")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--name", default="", help="评估目录名，默认自动生成")
    p.add_argument("--tag", default="", help="展示用标签，会拼进目录名")
    p.add_argument("--evals-dir", default="", help="评估产物根目录，默认 backend/storage/evals")
    p.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("export-model", help="把权重导出为部署格式（ONNX / TorchScript 等）")
    p.add_argument("--weights", default="", help="权重文件路径")
    p.add_argument("--formats", default="onnx,torchscript", help="逗号分隔的格式，如 onnx,torchscript")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="", help="cpu | cuda:0；默认读 YOLO_STUDIO_DEVICE")
    p.add_argument("--half", action="store_true", help="FP16（需要 GPU 支持时有意义）")
    p.add_argument("--dynamic", action="store_true", help="动态输入尺寸")
    p.add_argument("--no-simplify", action="store_true", help="关闭 ONNX 图简化")
    p.add_argument("--opset", type=int, default=0, help="ONNX opset，0 = 默认")
    p.add_argument("--int8", action="store_true", help="INT8 量化（部分格式）")
    p.add_argument("--nms", action="store_true", help="把 NMS 并入导出图（部分格式）")
    p.add_argument("--no-copy", action="store_true", help="不把产物复制到导出目录")
    p.add_argument("--name", default="", help="导出目录名，默认自动生成")
    p.add_argument("--tag", default="", help="展示用标签，会拼进目录名")
    p.add_argument("--deploys-dir", default="", help="导出根目录，默认 backend/storage/deploys")
    p.add_argument("--list-formats", action="store_true", help="只列出格式与本机可用性")
    p.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    p.set_defaults(func=_cmd_export_model)

    p = sub.add_parser("infer", help="用权重对图片推理，验证识别情况（M6 实时验证的命令行版）")
    p.add_argument("images", nargs="*", help="一个或多个图片路径")
    p.add_argument("--weights", default="", help="权重文件路径（.pt / .onnx / ...）")
    p.add_argument("--classes", default="", help="逗号分隔的类名；ONNX 等不带类名的格式必须给")
    p.add_argument("--task", default="", help="detect | classify | segment；空则由模型决定")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cpu | cuda:0；默认读 YOLO_STUDIO_DEVICE")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--max-det", type=int, default=300, help="最多保留多少个框")
    p.add_argument("--save", default="", help="把带框的结果图保存到该目录")
    p.add_argument("--list-formats", action="store_true", help="只列出可推理格式与本机可用性")
    p.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    p.set_defaults(func=_cmd_infer)

    p = sub.add_parser("prelabel", help="用权重批量预标注图片目录，产出伪标签（可用 --out 直接导出数据集）")
    p.add_argument("--weights", default="", help="权重文件路径（.pt / .onnx / ...）")
    p.add_argument("--images", default="", help="待标注图片目录（会递归收集）")
    p.add_argument("--classes", default="", help="逗号分隔的类名；ONNX 等不带类名的格式必须给")
    p.add_argument("--task", default="detect", choices=["detect", "classify", "segment"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cpu | cuda:0；默认读 YOLO_STUDIO_DEVICE")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--max-det", type=int, default=300, help="每张图最多保留多少个框")
    p.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 张（预览用）")
    p.add_argument("--out", default="", help="导出目录（目录名会自动带 prelabel 前缀）")
    p.add_argument("--split-ratio", default="0.8,0.1,0.1", help="导出时的 train,val,test 比例")
    p.add_argument("--seed", type=int, default=42, help="划分随机种子")
    p.add_argument("--no-stratify", action="store_true", help="关闭分层划分")
    p.add_argument("--no-groups", action="store_true", help="不按 group 分组（视频数据有泄漏风险）")
    p.add_argument("--no-respect-split", action="store_true", help="忽略输入里已有的划分")
    p.add_argument("--no-split", action="store_true", help="导出时不重新划分")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖非空导出目录")
    p.add_argument("--list-formats", action="store_true", help="只列出可推理格式与本机可用性")
    p.add_argument("--json", action="store_true", help="以 JSON 输出预标注报告")
    p.set_defaults(func=_cmd_prelabel)

    return parser


def _force_utf8_console() -> None:
    """Windows 控制台默认 GBK，会导致中文输出乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if getattr(stream, "encoding", "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
