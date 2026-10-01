# 交接说明（给新的会话/协作者）

本文件用于让**全新会话**在零上下文的情况下继续本项目。
配套阅读：[`PROGRESS.md`](../PROGRESS.md)（进度与决策）、[`README.md`](../README.md)（架构）、[`docs/getting-started.md`](getting-started.md)（运行方式）。

---

## 一、当前状态

| 阶段 | 进度 | 说明 |
|---|---|---|
| M0 项目骨架 | 5/5 | 完成 |
| M1 数据模块 | 12/12 | 完成 |
| M2 训练模块 | 7/7 | 完成 |
| M3 评估与模型库 | 4/4 | 完成 |
| **M4 部署导出** | **3/3** | **完成** |
| M5 前端界面 | **11/11** | **完成**（含 M5-09 路由级分割、M5-10 ECharts 按需引入、M5-11 设置页） |
| M6 实时验证 | **8/8** | **完成**（含前端页面，已真实浏览器验证） |
| M7 预标注 | **3/3** | **完成**（core / API / 前端向导 / CLI / 测试 / 血缘） |

**测试现状：1004 断言**（11 个测试文件，见第四节）。其余 10 个文件（含 `test_prelabel` 97）**稳定全绿**；
`test_train.py` 存在一个**环境相关的偶发失败**（单跑约 1/8~1/12，全量连跑约 1/3，**与 M7 无关**，见 R-47）——
它会让「全量一次跑绿」变成概率事件，别据此怀疑自己刚改的东西。

**下一步任务**：M0~M7 全部完成，完整闭环（数据 → 训练 → 评估 → 模型库 → 导出 → 实时验证 → 预标注）。
后续可选：M6 视频模式逐帧送检（当前有意未做）；在有浏览器的环境复核 M5/M7 页面交互（R-33）；
继续压缩 vendor chunk（R-32）；需要时安装 openvino 等可选依赖解锁更多格式。

> M2 + M3 + M4 + M5 已可用：训练 → 自动评估 → 模型注册 → 多模型对比 → 导出部署（含产物校验），
> 且全部有可视化界面：训练列表/新建、训练详情（实时曲线 + 资源 + 日志）、模型库/对比/导出、数据浏览器。

---

## 二、协作约定（重要，用户明确要求过）

这些偏好是在实际协作中确定的，新会话请遵守：

1. **不要在代码或文档里写"自我论证式"注释**。
   用户原话："也别专门强调本项目自己的 .venv 之类的注释"。
   反例：`# 本项目自包含，不依赖任何外部项目`、`# 这是可选灵活性，不是必需依赖`。
   正例：`# 训练使用的解释器，默认当前解释器。需要跑在别的环境时用 X 覆盖。`

2. **中文注释与中文文档**。界面文案也用中文。

3. **前后端严格分离**：`backend/` 与 `frontend/` 独立，通过 HTTP/WebSocket 通信。
   后端内部再分层：`core`（纯逻辑，不依赖 FastAPI）← `app`（FastAPI）← `cli`。

4. **进度必须落到 `PROGRESS.md`**：每完成一个任务就更新状态，并追加变更记录。

5. **验证必须真实执行并如实汇报**。
   不要用"应该可以"代替实际运行；失败就报失败；发现之前的结论有误要**明确更正**（例如曾误报"11 个越界框"，复核后是浮点噪声，已在 `PROGRESS.md` 更正）。

6. **不编造指标**。例如质量报告刻意**不做**"数据质量评分"——那种分数没有可验证含义。
   只报告能从数据算出的事实。

7. **改数据要留后路**：清洗/规范化只作用于内存中的 IR，不修改原始数据文件；
   处置前有 dry-run；删除都要可回溯。

8. **不要动用户其它项目**（`train_dms_model`、`DMD-Driver-Monitoring-Dataset` 等只作只读参考）。

---

## 三、环境事实

| 项 | 值 |
|---|---|
| 项目路径 | `D:\phpstudy_pro\WWW\yolo-studio` |
| 后端 venv | `backend/.venv`（Python 3.9.25，uv 创建） |
| 后端端口 | **8010**（8000 被本机 LLM 网关占用） |
| 已装依赖 | ultralytics 8.4.160 + torch 2.8.0（**CPU 版，无 CUDA**）；psutil 7.2.2；pynvml 已装但无 NVIDIA 设备 |
| 前端 | React 18 + Ant Design 5 + Vite + ECharts，端口 5173 |
| 启动 | `powershell -ExecutionPolicy Bypass -File scripts\dev.ps1` |

**本机无 CUDA** → 训练以 CPU 运行；可视化在无 GPU 时降级为仅 CPU 面板（已实现，见 M2-05）。

**未安装的可选依赖**（影响能力探测结果，不是 bug）：`openvino` /
`tensorflow` / `coremltools` / `rknn-toolkit2` 均未安装。
**已安装**：`onnx` + `onnxslim`（ONNX **导出**可用）、`onnxruntime==1.19.2`（ONNX **推理**可用，M6-07）、
`httpx`（测试用，FastAPI TestClient 依赖它）、`cv2`（仅 CLI 的 `infer --save` 用到，M6 方案 A 不需要）。
`pandas` 未安装（ultralytics 的 `export_formats()` 用不了，因此本项目自建格式清单，不依赖它）。

**当前后端状态**：已在本机 8010 端口运行（`uvicorn app.main:app`，非 `--reload`）。
改完后端代码需重启进程才会生效：
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*uvicorn*app.main*' } | Stop-Process -Force
cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

**现成可看的真实产物**（可直接用来调前端，不必重新跑训练）：

| 目录 | 内容 |
|---|---|
| `backend/storage/runs/smoke_yolo11n` | 1 epoch 真实训练（1 轮指标、12 张过程图） |
| `backend/storage/runs/smoke_e3` | 3 epoch 真实训练（3 轮指标、13 张过程图） |
| `backend/storage/models/{smoke_yolo11n,smoke_e3}` | 模型卡片（含评估索引与导出索引） |
| `backend/storage/evals/` | 2 次真实评估（逐类指标、混淆矩阵、曲线图） |
| `backend/storage/deploys/` | 1 次真实 TorchScript 导出（10.3 MB + sha256） |

> 这些是 12 张图的冒烟模型，mAP 为 0 属正常，**不代表任何模型质量结论**。

**真实训练已验证可用**（非模拟）：`yolo11n.yaml` 从零、CPU、imgsz=32、1 epoch，退出码 0 并产出
`results.csv` / `train.log` / 12 张过程图 / best.pt。权重用 `.yaml` 结构文件从零训练，**不需要联网下载**。
看现成产物：`backend/storage/runs/smoke_yolo11n/`。

---

## 前端现状（M5 已全部完成）

```
frontend/src/
├── main.tsx / App.tsx          路由表（全部页面 React.lazy，按页拆 chunk）
├── layouts/MainLayout.tsx      侧边栏菜单 + Suspense 兜底
├── api/client.ts               axios 封装 + 各接口方法 + wsUrl()
├── types.ts                    与后端 core/* 对应的 TS 类型
├── utils.ts                    formatBytes / formatDuration / formatMetric / formatTime
├── hooks/useJobSocket.ts       训练/评估/导出的 WS 订阅（4404 不重连，其余 2s 重连）
├── pages/Dashboard.tsx         总览
├── pages/DatasetImport.tsx     导入向导
├── pages/DatasetBrowser.tsx    M5-04 数据浏览器（筛选 + 分页 + 标注叠加）
├── pages/DatasetVersions.tsx   数据集版本与血缘
├── pages/QualityReport.tsx     质量报告（ECharts）
├── pages/TrainJobs.tsx         M5-07 训练列表
├── pages/TrainNew.tsx          M5-07 新建训练
├── pages/TrainDetail.tsx       M5-05 训练详情（WS 实时）
├── pages/ModelLibrary.tsx      M5-08 模型库列表（多选对比）
├── pages/ModelDetail.tsx       M5-08 模型详情（概览 / 评估 / 导出部署）
├── pages/ModelCompare.tsx      M5-08 模型对比
├── pages/VerifyCenter.tsx      M6 实时验证（摄像头 / 图片 / 视频切换 + 叠加）
├── pages/PrelabelWizard.tsx    M7 预标注向导（权重 + 图片 → 预览复核 → 生成数据集）
├── pages/Settings.tsx          M5-11 设置（只读配置与环境，数据来自 /api/system/config）
└── components/                 SampleGrid / ExportStep / StatusTag / MetricsChart /
                                ResourcePanel / LogViewer / ArtifactGallery / EvalResultView
```

### 前端要注意的坑

1. **WS 的 `progress` 字段在两类事件里类型不同**（已在前端类型里区分）：
   `metrics` 事件的 `progress` 是结构体 `{epoch, epochs_done, epochs_total, percent}`，
   而 `status` 事件的 `progress` 是 0~1 的数字（直接来自 `job.progress`）。别当成同一个东西。
2. **开发环境 WS 需要 Vite 代理开 `ws: true`**（`vite.config.ts` 已加）。
   否则 `/api/*` 的普通请求能通，WebSocket 却连不上。
3. **`finished` 事件到达时后端已完成联动**（注册模型 / 挂评估结果 / 挂导出产物），
   前端收到后直接刷新即可，不需要 sleep 或重试。
4. **缺失指标显示为空而不是 0**：逐类对比表、模型卡片都可能缺某项，前端一律留空/占位符。
5. **代码分割**：新增页面记得在 `App.tsx` 里用 `lazy(() => import(...))` 注册，保持一致。
6. **ECharts 按需引入**：图表统一从 `components/echarts.tsx` 导入（不是 `echarts-for-react`）。
   该文件用 `echarts/core` 只注册了 line / bar / heatmap 与必要组件；**新增其它图表类型
   （饼图、散点、雷达等）时必须先在 `echarts.tsx` 里 `echarts.use([...])`**，否则运行时报未注册。
7. **不要用 `manualChunks` 强制把 antd 打成一个 chunk**：实测会变成 ~1.1MB 且首屏全量加载，
   比 Rollup 按页面自然分包更差。当前 echarts 约 576 kB、antd 核心约 628 kB，构建有 >500KB 警告但不影响运行。

---

## 四、如何验证（改动后必跑）

```powershell
cd backend
.\.venv\Scripts\python.exe tests\test_ingest.py      # 63  接入层（含 /api/datasets/browse）
.\.venv\Scripts\python.exe tests\test_pipeline.py    # 89  划分 + 导出 + 多源合并
.\.venv\Scripts\python.exe tests\test_clean.py       # 58  清洗引擎
.\.venv\Scripts\python.exe tests\test_taxonomy.py    # 69  类别规范化
.\.venv\Scripts\python.exe tests\test_adapters.py    # 55  COCO / VOC / LabelMe
.\.venv\Scripts\python.exe tests\test_analytics.py   # 84  统计分析
.\.venv\Scripts\python.exe tests\test_train.py       # 142 训练模块（调度/指标/日志/资源/产物/API/WS）
.\.venv\Scripts\python.exe tests\test_eval.py        # 147 评估与模型库（结果归一化/报告/注册/对比/API）
.\.venv\Scripts\python.exe tests\test_deploy.py      # 103 部署导出（格式探测/哈希校验/生命周期/API/联动）
.\.venv\Scripts\python.exe tests\test_infer.py       # 97  M6 实时验证（格式/归一化/会话/协议/API/背压/类名来源）
.\.venv\Scripts\python.exe tests\test_prelabel.py    # 97  M7 预标注（收集/IR/统计/失败跳过/分类/血缘/导出卡片/API）
```

> Windows 控制台是 GBK，含 `²` 等字符的输出会抛 `UnicodeEncodeError`。
> 遇到时先设 `$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'` 再跑（这不是测试失败）。

**测试夹具**：
- `backend/tests/fixtures/dmd_sample/`（60 帧 + VCD，由官方 `vcd` 库生成，已提交）
- COCO / VOC / LabelMe 夹具由 `test_adapters.py` 程序化生成，无需额外文件
- `backend/tests/fixtures/fake_trainer.py`：模拟 ultralytics 外部行为的假训练进程（写 results.csv / 日志 / 过程图），用于快速验证调度，无需真跑 YOLO
- `backend/tests/fixtures/make_tiny_det.py`：生成极小检测数据集（8 train + 4 val + 2 test），供**真实训练**与**真实评估**冒烟用；产物落在 `backend/storage/demo_sources/tiny_det`（storage 目录被 .gitignore 忽略）
- `backend/tests/fixtures/fake_evaluator.py`：模拟 ultralytics val 的假评估进程（写 eval_result.json / 混淆矩阵 / 曲线图），用于快速验证调度与模型库
- `backend/tests/fixtures/fake_exporter.py`：模拟 ultralytics export 的假导出进程（按格式产出文件 + deploy_result.json），可制造部分成功/全部失败

**真实训练怎么跑**（不需要联网，权重用包内 `.yaml` 从零构建）：

```powershell
cd backend
.\.venv\Scripts\python.exe tests\fixtures\make_tiny_det.py
.\.venv\Scripts\python.exe -m cli.main train --data storage\demo_sources\tiny_det\data.yaml `
  --weights yolo11n.yaml --epochs 1 --imgsz 32 --batch 4 --device cpu --name smoke_yolo11n
# 只组装命令、不启动训练：加 --dry-run
```

前端：`cd frontend; npm run build`（含 TypeScript 类型检查）。

**真实评估怎么跑**（先完成上面的真实训练）：

```powershell
# 命令行直接评估（不依赖 Web）
.\.venv\Scripts\python.exe -m cli.main eval `
  --weights storage\runs\smoke_yolo11n\weights\best.pt `
  --data storage\demo_sources\tiny_det\data.yaml --split test --imgsz 32 --device cpu

# 或用 API：POST /api/models/register {"job_id":"smoke_yolo11n"}
#          POST /api/models/smoke_yolo11n/eval {"split":"test"}
#          GET  /api/models 查看，POST /api/models/compare 对比
#          POST /api/eval/jobs/{eval_id}/report 生成单文件 HTML 报告
```

**真实导出怎么跑**（M4，本机只装了 torch，因此只有 torchscript 可用）：

```powershell
.\.venv\Scripts\python.exe -m cli.main export-model --list-formats
.\.venv\Scripts\python.exe -m cli.main export-model `
  --weights storage\runs\smoke_e3\weights\best.pt --formats torchscript --imgsz 32 --device cpu
# 想真跑 ONNX：uv pip install onnx onnxslim（可选，见 getting-started）
```

---

## 五、真实数据的情况（用于回归验证）

| 数据 | 路径 | 特征 |
|---|---|---|
| DMS 检测数据集 | `D:\phpstudy_pro\WWW\train_dms_model\data\archive` | 9884 图 / 23389 框 / 5 类；**自带 Roboflow 划分且存在类别构成偏移** |
| DMD 分类数据集 | 仓库 `DMD-Driver-Monitoring-Dataset`（仅工具，无数据） | 夹具已模拟 |

**已查明的数据问题**（回归时可用作对照）：
- 原始 DMS 数据集有 **115 对跨子集完全重复** + 220 对跨子集近似重复 → 验证指标虚高
- 视频帧文件名形如 `-s4-93-_mp4-75_...`，可用 `--group-by "regex:^(.*?)_mp4-\d+"` 提取 **114 个视频组**
- 按视频分组重新划分后：同组跨子集 **0**，近似重复泄漏降到 81 对（残余属跨录制场次固有相似）
- test 集类别构成偏移：Cigarette 32%（整体 17%）、Open Eye 36%（整体 48%）

---

## 六、可直接粘贴的续作提示

```
继续开发 D:\phpstudy_pro\WWW\yolo-studio。

先读这三个文件建立上下文：
- docs/HANDOFF.md（协作约定与环境，务必遵守）
- PROGRESS.md（进度、已决策事项、风险项）
- README.md（架构与设计决策）

M0~M7 已完成：M1 数据模块（12/12）、M2 训练（7/7）、M3 评估与模型库（4/4）、
M4 部署导出（3/3）、M5 前端界面（11/11）、M6 实时验证（8/8）、M7 预标注（3/3）。
测试 1004 断言（11 个文件；`test_train` 有环境相关偶发，见 R-47），前端 `npm run build` 通过。

后续可做的事（按需选择，先读 PROGRESS.md 风险表）：
- M6 视频模式的逐帧送检（当前有意未做）；
- 在有浏览器的环境复核 M5 / M7 页面交互（R-33）；
- 用 manualChunks / echarts 按需引入继续压缩 vendor chunk（R-32）；
- 需要时安装 openvino 等可选依赖，解锁更多导出/推理格式。
  （onnx / onnxslim / onnxruntime 已装，ONNX 导出与推理均已真实跑通。）

动手前先跑一遍测试确认基线，每完成一个任务更新 PROGRESS.md（含验证记录与变更记录）。

注意：本机无 CUDA（CPU 训练），可视化需支持无 GPU 降级。
训练/评估/导出相关坑见 HANDOFF.md 第七、八、九节；M6 坑见第十节；M7 坑见第十一节；前端坑见「前端现状」小节。
```

---

## 七、M2 训练模块的文件地图与坑（避免重复决策）

上一轮已确定、新会话不必再讨论的：

1. **训练框架范围**：一期仅 ultralytics，但保留 `TrainerBackend` 抽象
   （`build_command()` / `parse_metrics()` / `resolve_weights()`），便于将来接 YOLOv5 或 MMDetection。
2. **指标来源**：tail 训练目录下的 `results.csv`，**不 hook ultralytics 内部**。
   这样升级 ultralytics 不会把可视化搞坏。
3. **进程隔离**：训练通过 subprocess 启动，崩溃不影响 API 进程。
4. **训练解释器**：默认当前解释器，可用 `YOLO_STUDIO_PYTHON` 覆盖；设备用 `YOLO_STUDIO_DEVICE`。
5. **推送方式**：WebSocket 增量推送（不是前端轮询）。

### 文件地图

```
core/train/
├── spec.py                  TrainSpec：core 与子进程之间的唯一参数契约
├── backend.py               TrainerBackend 抽象（三个纯函数钩子）
├── ultralytics_backend.py   ultralytics 实现；train() 是唯一 import ultralytics 的地方
├── runner.py                训练子进程入口：python -m core.train.runner --spec <json>
├── metrics.py               results.csv 解析（增量 / 去重 / 半行容错 / 检测+分类列集）
├── job.py                   状态机取值 + TrainingJob 可持久化记录（job.json）
├── manager.py               TrainingManager：子进程生命周期、监控线程、事件订阅、接管恢复
├── resources.py             CPU(psutil) / GPU(pynvml) 采样，无 GPU 降级；resolve_device()
└── artifacts.py             过程图像分组 + 目录穿越防护
app/api/routes/train.py      /api/train/*（REST + WebSocket）
app/schemas.py               TrainRequest / TrainJobResponse 等
tests/test_train.py          142 断言
tests/fixtures/fake_trainer.py   假训练进程
tests/fixtures/make_tiny_det.py  真训练冒烟数据集生成器
```

### 两个必须知道的坑（已修复，改动前先读）

1. **`project` 必须传绝对路径**（R-19）。
   ultralytics 对相对 `project` 会拼上它自己的 `SETTINGS["runs_dir"]`，
   实测结果写到了 `backend\runs\detect\storage\runs\<job>`，导致 `results.csv` 找不到。
   `TrainingManager._apply_defaults` 已强制 `Path(project).resolve()`，
   新增调用方不要绕过它。

2. **`data.yaml` 的 `path` 也建议绝对路径**（R-20）。
   ultralytics 解析相对 `path` 的基准不是 yaml 所在目录。本项目导出器写的是绝对路径；
   手工数据集请照 `tests/fixtures/make_tiny_det.py` 写绝对 `path`。

### 其它约定

- 训练日志同时进内存环形缓冲与 `run_dir/train.log`；服务重启后从文件恢复历史。
- 接管仍在运行的训练进程时，重启前的 stdout 无法补记（只能继续读 results.csv），
  这是有意的取舍，不要为此去改 ultralytics。
- 已结束任务的日志不会被当作增量重复推送（`_load_log_history` 里会置位 `emitted_log_seq`）。
- **终态必须最后置位**：`_finalize` 先刷新指标/日志，最后才写 `job.status`。
  否则外部一看到 `finished` 就去读指标，可能读到缺最后几轮的结果。
- **上层联动要在 `finished` 事件之前完成**：`_notify_finish` 现在排在 `_emit("finished")` 之前。
  否则客户端收到 `finished` 立刻取模型卡片，会读到还没挂上评估结果的旧卡片。
- 推理与调度管理器都提供 `refresh()`，`list()`/`get()` 会自动补扫目录——
  服务启动后用 CLI 新增的任务才能被 API 看到。
- 测试必须隔离：训练/评估用例注入隔离的 registry 并关闭自动评估，
  否则「训练完成自动注册」会把卡片写进真实 `storage/`。

---

## 八、M3 评估与模型库的文件地图

```
core/eval/
├── spec.py              EvalSpec（评估参数契约）
├── result.py            EvalResult / ClassMetrics + build_result()（纯函数归一化）
├── ultralytics_eval.py  extract_metrics()（唯一 import ultralytics 处）+ run()
├── runner.py            评估子进程入口：python -m core.eval.runner --spec <json>
├── manager.py           EvalManager：子进程调度、日志、接管、结果读取、on_finish 回调
├── report.py            单文件 HTML 评估报告（内联 SVG + base64 内嵌 PNG）
└── compare.py           多模型对比（同划分校验、缺失留空）
core/registry/
├── model_card.py        ModelCard（model_card.json）
└── registry.py          ModelRegistry：注册 / 挂评估 / 数据集血缘反查
core/process.py          pid_alive / kill_tree（训练与评估共用）
app/services.py          训练/评估/模型库三个单例 + 「训练完成 → 注册 + 自动评估」「评估完成 → 挂卡片」联动
app/api/routes/eval.py   /api/eval/*
app/api/routes/models.py /api/models/*
```

存储布局：

```
storage/evals/<eval_id>/     eval_spec.json / eval_result.json / eval.log + ultralytics 过程图
storage/models/<job_id>/     model_card.json（模型 id 就是训练任务 id）
```

### M3 的两个坑

1. **numpy 数组不能用 `or`**（R-23）。
   `Metric.p` / `ap_class_index` / `nt_per_class` 都是 ndarray，
   `getattr(x, "p", []) or []` 会抛「truth value of an array is ambiguous」。
   统一用 `core/eval/ultralytics_eval._as_list()`。
   这条路径只测 `build_result` 覆盖不到，必须让真实评估跑一次。

2. **不同划分的数字不可比**（M3-04）。
   对比时若各模型的评估划分不一致，`compare_eval_results` 会在 `notes` 里明确提示，
   不要把它当成 bug 去掉——那是在防止错误结论。

### 自动联动（不要重复实现）

- 训练成功 → `services._after_training_finish`：注册模型；若 `YOLO_STUDIO_AUTO_EVAL` 非 0 则自动评估
  （划分取 `YOLO_STUDIO_AUTO_EVAL_SPLIT`，默认 `auto` = 有 test 用 test，否则 val）。
- 评估结束 → `services._after_eval_finish`：把结果挂到 `spec.job_id` 对应的模型卡片。
- 导出结束 → `services._after_deploy_finish`：把产物挂到 `spec.job_id` 对应的模型卡片。
- 三个回调都由 `services.set_managers()` 统一注册（注入管理器时也会挂上），
  测试里不要自己再注册一遍。

---

## 九、M4 部署导出的文件地图

```
core/deploy/
├── formats.py          格式清单 + 能力探测（find_spec，不 import 重依赖）
├── spec.py             DeploySpec（导出参数契约）
├── result.py           Artifact / DeployResult / sha256 / verify_result
├── ultralytics_export.py  export_kwargs()（纯函数）+ run()（唯一 import ultralytics 处）
├── runner.py           导出子进程入口：python -m core.deploy.runner --spec <json>
└── manager.py          DeployManager：调度、按格式逐个导出、产物复制与哈希、接管
app/api/routes/deploy.py  /api/deploy/*
```

存储布局：

```
storage/deploys/<deploy_id>/   deploy_spec.json / deploy_result.json / deploy.log + 复制过来的产物
```

### M4 的关键约定

- **格式可用性靠 `find_spec` 判断**（R-27），别改成真正 import——
  那会把 onnx/openvino 拉进 API 进程，启动变慢且可能因子包副作用出错。
- **不可用的格式必须在 `start()` 前拒绝**，并说明缺哪个包 / 是否需要 GPU。
  启动一个注定失败的子进程只会让用户等一场空。
- **每个格式独立成败**：有失败格式也没关系，只要有一个成功就算 `finished`；
  失败原因记在对应 `Artifact.error` 上。
- **产物校验只比文件完整性**（存在 / 大小 / sha256），不重新推理（R-29）。
  大小不一致时不再算哈希（省一次全文件读取）。
- **测试里的假导出后端**要注入 `format_checker`，否则本机缺包会导致 `onnx` 等格式被拒绝。
- `Manager` 的 `on_finish` 回调排在 `finished` 事件之前（与训练/评估一致）。

---

## 十、M6 实时验证的文件地图（后端已完成，前端待做）

设计文档：`docs/m6-realtime-verify.md`（定位、方案取舍、风险，动手前先读）。

```
core/infer/
├── formats.py            推理格式清单 + 能力探测（find_spec，不 import 重依赖）
├── spec.py               InferSpec（会话级：权重/类名/设备/imgsz）+ InferOptions（帧级：conf/iou）
├── result.py             FrameResult / Detection + build_frame_result()（纯函数归一化）
├── ultralytics_infer.py  真正的推理（唯一 import ultralytics/torch 处）
├── worker.py             常驻推理子进程入口：python -m core.infer.worker（JSON Lines 协议）
└── session.py            InferSession：主进程侧的子进程生命周期 + 请求-响应配对
app/services_infer.py     推理会话单例 + 权重来源发现（模型库 + 导出产物）
app/api/routes/infer.py   /api/infer/*
tests/test_infer.py       89 断言
cli 的 infer 子命令        命令行推理（--list-formats / 多图 / --json / --save）
```

### M6 的关键约定（改前必读）

1. **这是常驻子进程，不是一次性任务**。训练/评估/导出都是「起进程→跑完→退出」，
   M6 要反复喂帧，逐帧起进程要几秒（import torch），不可用。因此走
   stdin/stdout 的 **JSON Lines** 协议。协议内容见 `worker.py` 顶部注释。
2. **会话不是线程安全的**（R-39）。`InferSession` 用共享字典做请求-响应配对，
   必须串行调用：API 层用 `services_infer.session_lock()`，WS 用 `pending/processing` 状态机。
3. **崩溃感知不能只靠读线程**（R-38）。Windows 上 `for line in proc.stdout` 不保证在
   子进程退出时立即返回，等待循环要主动 `poll()`（`InferSession._check_dead`）+ 有界 `wait(0.5s)`。
   否则「子进程崩溃」会被误报成「超时」。
4. **帧率上限是设计取舍**（R-35）。方案 A（浏览器抓帧→WS→后端推理）在 CPU 上约 3~15 FPS，
   够「肉眼验证」不够流畅视频。前端必须做**丢帧**防延迟堆积——不是 bug，不要试图靠排队解决。
5. **ONNX 的类名不一定有**。ultralytics 导出时通常会把 names 写进 ONNX metadata（`YOLO()` 能读出），
   但**不保证**——没有时必须在请求里给 `classes`，否则只能显示 `class_0`。
   类名的解析必须走 `ultralytics_infer.resolve_names()`，**不要**在别处再读一次 `model.names`
   （R-41 就是这么踩出来的：加载时显示一套类名、框上是另一套）。
6. **M6 不落盘、不产数据集**。它是「显示」，伪标注是 M7 的事（见 R-37，必须人工复核）。
7. **前端页面已完成并真实浏览器验证**（摄像头出画面、叠加生效）。**视频模式的逐帧送检未实现**——
   页面上如实写了，不要留「点了没反应的假按钮」。

---

## 十一、M7 预标注的文件地图与坑

定位见 `PROGRESS.md` 的 M7 章节与 **R-37**：产出的是**伪标签**，不是标注，必须人工复核。

```
core/prelabel/
├── prelabel.py            list_images / build_bundle / annotate_bundle / run_prelabel
│                          + attach_lineage / ensure_prelabel_prefix / export_name
app/services_prelabel.py   PrelabelJob 后台线程任务（独立 InferSession）+ export_job 复用现成流水线
app/api/routes/prelabel.py /api/prelabel/*（formats/weights/jobs/samples/export/delete）
app/schemas.py             PrelabelRequest / PrelabelJobResponse / PrelabelExportRequest ...
frontend/src/pages/PrelabelWizard.tsx   三步向导（配置 → 预览 → 生成数据集）
tests/test_prelabel.py     97 断言（假 worker 脚本，不依赖 torch）
cli/main.py 的 prelabel    命令行预标注（--out 时顺手导出）
```

### M7 的关键约定

1. **只做「推理结果 → IR」**。`core/prelabel` 只把 `FrameResult` 写进 `DatasetBundle`；
   类别规范化 / 清洗 / 划分 / 导出一律调用现成的（`services_prelabel.export_job`），不要重写。
2. **推理复用 `core/infer`**，但预标注任务用**自己的** `InferSession`（与 M6 实时验证的会话分开），
   避免把实时验证正在用的模型换掉。任务跑在后台线程，前端轮询，不阻塞 HTTP。
3. **单张失败不中断整批**：`InferError`（崩溃/超时）与 worker 业务错误（`ok=false`，如图像损坏）
   都记入 `report.failures` 并跳过。**不要把业务错误当成「这张图没有目标」**——那会把失败藏进空标注。
4. **`boxes_total` 必须累计**（曾因未累加而报告 0 框、实际 21 个）。分类任务按整图标注计数。
5. **血缘固定落在 `dataset_card.json` 的 `prelabel` 段**（权重 / conf / iou / 模型类别 / 生成时间 + 免责说明），
   非预标注数据集该键为 `null`。导出目录名强制 `prelabel` 前缀。
6. **任务存在内存里**（R-45）：服务重启即丢，重跑即可；界面已注明「仅本次后端进程内」。
