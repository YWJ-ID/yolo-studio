# 项目进度追踪

> **唯一事实来源**。每完成一个任务，必须同步更新本文件的状态与日期。
> 状态取值：`未开始` / `进行中` / `已完成` / `阻塞` / `已放弃`
> 更新规则：状态变更时同时更新「最近更新」日期，并在「变更记录」追加一行。

- 项目：YOLO Studio
- 技术栈：FastAPI (Python 3.9) + React 18 + Ant Design 5 + ECharts
- 数据存储：文件系统（数据集版本自带的 `dataset_card.json` 记录血缘，无需数据库）
- 训练后端：ultralytics（一期），预留 `TrainerBackend` 抽象
- 最近更新：2026-09-30

> 新会话/新协作者请先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)：
> 里面有协作约定、环境事实、验证方式与可直接粘贴的续作提示。

---

## 总览

| 阶段 | 模块 | 进度 | 状态 |
|---|---|---|---|
| M0 | 项目骨架 | 5/5 | 已完成 |
| M1 | 数据模块 | 12/12 | **已完成** |
| M2 | 训练模块 | 7/7 | **已完成** |
| M3 | 评估与模型库 | 4/4 | **已完成** |
| M4 | 部署导出 | 3/3 | **已完成** |
| M5 | 前端界面 | 11/11 | **已完成** |
| M6 | 实时验证 | 8/8 | **已完成**（含前端页面，已真实浏览器验证） |
| M7 | 预标注（模型批量标框） | 3/3 | **已完成**：core / API / 前端向导 / CLI / 测试 / 血缘 |

**MVP 定义（一期目标）**：M0 全部 + M1 的 `M1-01/02/03/08/09/10` + M2 的 `M2-01/02/03/04` + M5 的 `M5-01/02/05`。

---

## M0 项目骨架

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M0-01 | 仓库结构与文档（README / PROGRESS / docs） | 已完成 | README、PROGRESS、docs/getting-started.md |
| M0-02 | 后端 FastAPI 骨架 + 健康检查 | 已完成 | /api/system/health、/api/system/env |
| M0-03 | 前端 React + AntD + Vite 骨架 | 已完成 | 构建通过（tsc + vite build） |
| M0-04 | core 与 api 分层约束、配置管理 | 已完成 | `app/config.py`（全部路径 `YOLO_STUDIO_*` 可覆盖）；只读的前端设置页见 M5-11 |
| M0-05 | 一键启动脚本（dev 模式） | 已完成 | scripts/dev.ps1、start-backend/frontend.ps1 |

## M1 数据模块（Data Pipeline）

### 1.1 接入层

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M1-01 | Canonical IR 数据模型（ImageRecord / Annotation / DatasetBundle） | 已完成 | core/ir.py，实测 9884 图解析通过；支持 bbox / image 两种标注形态 |
| M1-02 | 接入适配器：YOLO txt + data.yaml | 已完成 | core/ingest/yolo.py，支持 4 种目录形态 |
| M1-03 | 接入适配器：COCO JSON | 已完成 | 多文件/单文件布局；bbox [x,y,w,h] 换算；按文件名与路径前缀识别划分 |
| M1-04 | 接入适配器：Pascal VOC XML | 已完成 | Annotations + JPEGImages + ImageSets/Main；`voc_one_based` 开关处理 1-based/0-based 歧义 |
| M1-05 | 接入适配器：LabelMe JSON | 已完成 | rectangle / polygon / circle；多边形保留 segmentation；line/point 跳过并告警 |
| M1-06 | 接入适配器：OpenLABEL / VCD JSON（DMD 数据集） | 已完成 | 逐帧动作 → 图像级标签；支持层级选择、object 并入、分组防泄漏 |

### 1.2 规范化与清洗

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M1-07 | 类别体系：别名映射 / 合并 / 拆分 / 校验 | 已完成 | 按类名归一化给合并建议（保守：不猜拼写）；keep/drop/重排/sanitize；**按标注形态过滤**（解决 R-09） |
| M1-08 | 清洗引擎：图像 / 重复 / 标签 / 泄漏 四类规则 | 已完成 | 16 条规则，dry-run + apply；近似重复用多重索引加速（2 万图 < 1s） |
| M1-09 | 数据集划分：随机 / 分层 / 分组（防泄漏） | 已完成 | 分组强制不拆散；全局修正最小样本量；报告各类别在 val/test 的缺失 |
| M1-10 | 导出 YOLO 目录结构 + data.yaml + dataset_card | 已完成 | 检测布局 + 分类布局；名称冲突处理；copy/hardlink/symlink |
| M1-11 | 数据集版本与血缘 | 已完成 | 每个数据集自带 `dataset_card.json`，版本列表 API + 页面；不引入 sqlite |
| M1-12 | 统计分析与质量报告（分布 / 抽样预览 / HTML） | 已完成 | 类别分布与不均衡量化、目标尺寸分档（对应 P3/P4/P5）、每图目标数、类别×子集矩阵、子集构成偏移检测；单文件 HTML 报告（内嵌 SVG + base64 缩略图） |

## M2 训练模块

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M2-01 | `TrainerBackend` 抽象 + ultralytics 实现 | 已完成 | `core/train/backend.py`（抽象：resolve_weights / build_command / parse_metrics）、`ultralytics_backend.py`；命令行经 `core.train.runner` 启动，API 进程不加载 torch/ultralytics |
| M2-02 | 训练任务调度与状态机（start/stop/resume） | 已完成 | `core/train/manager.py` + `job.py`；subprocess 隔离、进程树终止、断点续训；服务重启后接管仍存活的进程或标记 interrupted |
| M2-03 | 实时指标解析（tail results.csv） | 已完成 | `core/train/metrics.py`；不 hook ultralytics，按列名解析，支持增量、断点续训去重、半行容错、检测/分类两种列集 |
| M2-04 | WebSocket 实时推送 | 已完成 | `app/api/routes/train.py` 的 `/api/train/jobs/{id}/ws`；core 只暴露同步订阅回调，API 层用 `call_soon_threadsafe` 投递事件 |
| M2-05 | 资源监控（GPU pynvml / CPU psutil） | 已完成 | `core/train/resources.py`；无 NVIDIA 时降级为 `gpu.available=false + error`，CPU 数据始终可用；`resolve_device` 仅在 auto 时附带 import torch |
| M2-06 | 训练过程图像预览（增强图 / 预测图） | 已完成 | `core/train/artifacts.py`；按 val_pred / val_labels / train_batch / labels / curve / matrix 分组，只回传文件名，访问走受限接口 |
| M2-07 | 实时日志流（stdout → 网页终端） | 已完成 | stdout+stderr 合并读取（按 `\r`/`\n` 切分），内存环形缓冲 + `train.log` 落盘，重启后从文件恢复历史 |

## M3 评估与模型库

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M3-01 | 训练完成自动评估（val / test） | 已完成 | `core/eval/`（spec/result/manager/runner）；独立子进程跑 ultralytics val，结果落 `eval_result.json`；训练结束自动评估（`YOLO_STUDIO_AUTO_EVAL=0` 可关，划分默认 test 缺失则 val） |
| M3-02 | 指标表与图表（PR / F1 / 混淆矩阵） | 已完成 | 逐类指标表 + 混淆矩阵（含 background，标注方向）+ 各类 AP 条形图；ultralytics 的 PR/F1 曲线与混淆矩阵 PNG 按需内嵌，另有单文件 HTML 报告 `/api/eval/jobs/{id}/report` |
| M3-03 | 模型注册（关联数据集版本 + 训练配置） | 已完成 | `core/registry/`：每个模型一份 `model_card.json`（`storage/models/<job_id>/`），记录权重路径、训练超参、数据集血缘（反查 `dataset_card.json`，无卡片时读 data.yaml 的 names）与评估结果索引；训练完成自动注册 |
| M3-04 | 多模型 / 多实验对比 | 已完成 | `core/eval/compare.py` + `POST /api/models/compare`；总体指标表 + 逐类 AP50-95 对比；**不同划分会明确提示不具备可比性**，缺失指标留空不补 0 |

## M4 部署导出

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M4-01 | ONNX / TorchScript 导出 | 已完成 | `core/deploy/`（formats/spec/result/manager/runner）；导出跑在独立子进程；TorchScript 已真实跑通（10.27 MB），ONNX 因本机未装 `onnx` 被提前拒绝并说明原因 |
| M4-02 | RKNN 导出（对接 rknn_model_zoo） | 已完成 | 格式已注册（`requires=("rknn",)`、归 M4-02）；缺 `rknn-toolkit2` 时在**启动前**拒绝并给出可操作说明，不会启动注定失败的子进程 |
| M4-03 | 导出产物管理与校验 | 已完成 | 产物复制到导出目录并记录大小 + sha256 基线；`verify` 重新计算比对（文件不存在 / 大小不一致 / 内容已变化）；产物挂到模型卡片（`model_card.deploys`） |

## M5 前端界面

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M5-01 | 布局框架 / 路由 / 主题 | 已完成 | MainLayout + 7 条路由，未实现页有占位提示 |
| M5-02 | Dashboard 总览 | 已完成 | 运行环境、已接入格式、路线图 |
| M5-03 | 数据导入向导（分步 + 预览） | 已完成 | 扫描预览 → 加入合成清单（多格式合并）→ 划分预览 → 生成数据集 |
| M5-04 | 数据浏览器（缩略图 + 标注框叠加） | 已完成 | `pages/DatasetBrowser.tsx`；新增 `POST /api/datasets/browse`（分页 + 按类别/划分/来源/标注形态/目标数/图像长边/路径子串筛选）；缩略图复用 `SampleGrid` 叠加标注框 |
| M5-05 | 训练详情页（实时曲线 + 资源 + 日志） | 已完成 | `pages/TrainDetail.tsx` + `hooks/useJobSocket`；WS 增量（snapshot/metrics/log/resources/status/finished）；实时曲线（可切指标）、CPU/GPU 资源、网页终端日志、过程图像、参数与产物 |
| M5-06 | 数据集版本与质量报告页 | 已完成 | 版本与血缘页（DatasetVersions）+ 质量报告页（QualityReport，ECharts 图表 + HTML 导出） |
| M5-07 | 训练任务列表 / 新建训练页 | 已完成 | `pages/TrainJobs.tsx`（列表 + 停止/续训/注册，运行中自动轮询）、`pages/TrainNew.tsx`（数据集/权重/超参表单，无 CUDA 时给出 CPU 提示） |
| M5-08 | 模型库与对比页 | 已完成 | `pages/ModelLibrary.tsx`、`ModelDetail.tsx`（概览 / 评估 / 导出部署三个标签页）、`ModelCompare.tsx`；含发起评估、HTML 评估报告、导出格式可用性、产物下载与完整性校验 |
| M5-09 | 路由级代码分割（React.lazy） | 已完成 | `App.tsx` 全部页面改 `React.lazy` + `MainLayout` 内 `Suspense`；主包拆成按页 chunk |
| M5-10 | 前端产物优化（ECharts 按需引入） | 已完成 | `components/echarts.tsx` 用 `echarts/core` 只注册 line / bar / heatmap + 必要组件，替换全量 `echarts-for-react`；echarts chunk 由 1,052 kB 降到 576 kB（gzip 350→193 kB） |
| M5-11 | 设置页（只读配置与环境展示） | 已完成 | `pages/Settings.tsx` + 后端 `GET /api/system/config`；展示生效路径（含对应环境变量）、运行环境、训练设备、资源监控能力、导出格式可用性、可选依赖与安装命令、图片读取白名单；删除已无用的 `Placeholder.tsx` |

---

## M6 实时验证

> 设计文档：[`docs/m6-realtime-verify.md`](docs/m6-realtime-verify.md)
>
> 定位：拿已训练模型对**摄像头 / 图片 / 视频**跑推理并叠加显示，用肉眼验证识别情况。
> 与 M3 的区别：M3 有真值、算指标；M6 无真值、只显示。**不是第二个评估模块。**

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M6-01 | 权重格式探测 + `InferSpec` | 已完成 | `core/infer/formats.py`、`spec.py`；`find_spec` 探测，不 import 重依赖。7 种格式，`.pt`/`onnx`/`torchscript` 本机可用 |
| M6-02 | 常驻推理子进程 + 行协议 | 已完成 | `worker.py`（子进程入口）/ `session.py`（主进程侧）/ `ultralytics_infer.py`（唯一 import ultralytics 处）；JSON Lines 协议，含超时与崩溃感知 |
| M6-03 | 推理结果归一化（纯函数） | 已完成 | `result.py`，绝对像素 xyxy，与 IR 一致；NaN 不写成 0，越界下标合成类名不崩溃 |
| M6-04 | API：格式 / 权重 / 加载 / 单图推理 | 已完成 | `/api/infer/formats|weights|session|models|close|image` |
| M6-05 | WebSocket 实时流（背压 + 丢帧） | 已完成 | `/api/infer/ws`；推理走 `asyncio.to_thread` 不阻塞事件循环，处理中只保留最新一帧 |
| M6-06 | 前端验证页（摄像头 / 图片 / 视频 + 叠加） | 已完成 | `pages/VerifyCenter.tsx` + `/verify` 路由与菜单。**已真实浏览器验证**：摄像头出画面、逐帧推理（215 帧/6.3 FPS）、叠加生效。视频模式**有意未实现**（页面上如实说明，不做假按钮） |
| M6-07 | 安装并真实验证 ONNX 推理链路 | 已完成 | 已装 `onnxruntime==1.19.2`；**顺带正面验证了 M4 的 ONNX 导出**（此前从未真跑过） |
| M6-08 | CLI 子命令 `infer` | 已完成 | `cli.main infer`，支持 `--list-formats` / 多图 / `--json` / `--save` |

---

## M7 预标注（伪标签）

> **定位：预标注（pre-annotation），不是「自动标注」。**
>
> 模型只做**加速**，不做**替代**：预测框是**伪标签**，漏检会变成「没有这个标注」，
> 错检会被固化成标注。产出物须经**人工复核**后才可作为训练数据。
> dataset_card 里必须记录「框来自哪个权重、conf 阈值多少」，可追溯。
>
> 复用 M6 的 `core/infer/` 推理底座；额外需要把预测结果写进 **统一 IR**，
> 再复用已有的 taxonomy / clean / split / export。导出格式**暂不扩展**
> （现有 YOLO 检测 + YOLO 分类够用，COCO / VOC / LabelMe 按需再补）。

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| M7-01 | 批量推理 → 写入 IR（伪标注） | 已完成 | `core/prelabel/`：`build_bundle` / `annotate_bundle` / `run_prelabel`；复用 `core/infer` 常驻子进程；单张失败记入 `report.failures` 并跳过，不中断整批 |
| M7-02 | 预标注 API + 前端向导 | 已完成 | `/api/prelabel/*`（formats/weights/jobs/samples/export）；后台线程跑推理（CPU 批量耗时长，不阻塞 HTTP），前端轮询 + 三步向导 `pages/PrelabelWizard.tsx` |
| M7-03 | 血缘记录「伪标签」身份 | 已完成 | `dataset_card.json` 固定带 `prelabel` 段（权重/conf/iou/模型类别/生成时间 + 免责说明）；导出目录名强制 `prelabel` 前缀；页面与 CLI 均给出「必须人工复核」提示 |

**M7 补充交付（与其它模块保持一致）**：

| 项 | 状态 | 说明 |
|---|---|---|
| 测试 `backend/tests/test_prelabel.py` | 已完成 | 97 断言：list_images / build_bundle / annotate（正常·失败·崩溃·空目录·分类）/ 血缘 / 导出卡片 / API 路由与错误路径。**回归了历史统计 bug**（`boxes_total` 未累加） |
| CLI 子命令 `prelabel` | 已完成 | `--weights/--images/--conf/--iou/--limit/--out`；`--out` 时顺手导出（目录名自动带前缀）；`--list-formats` 复用推理格式探测 |

> **M7 的定位重申（R-37）**：产出的是**伪标签**，不是标注。漏检 = 缺标注、误检 = 错标注，
> 必须人工复核后才能作为训练数据。血缘与界面文案都不得暗示可跳过复核。

---

## 待决策 / 风险

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| R-01 | 训练框架范围 | 已决策 | 一期仅 ultralytics + `TrainerBackend` 抽象 |
| R-02 | 本机无 CUDA | 观察中 | 训练为 CPU 模式，可视化需支持无 GPU 降级 |
| R-03 | Python 版本 | 已决策 | Python 3.9（`backend/.venv`） |
| R-04 | 前端构建工具 | 已决策 | Vite |
| R-05 | 训练环境方案 | 已决策 | 训练依赖声明在 `requirements.txt` 随 venv 安装；训练解释器默认当前解释器，可用 `YOLO_STUDIO_PYTHON` 覆盖 |
| R-06 | 图片读取白名单 | 已实现 | 默认不限制（仅本机开发）；对外服务前必须设 `YOLO_STUDIO_ALLOWED_ROOTS` |
| R-07 | API 端口 | 已决策 | 用 **8010**。8000 在本机被 OpenAI 兼容网关（LibreChat/lobe-chat 类）占用，避免冲突 |
| R-08 | **DMD 数据是分类而非检测** | 已解决 | OpenLABEL/VCD 只有逐帧动作标签，**无边界框**。IR 支持 `kind=image`；导出已实现分类布局 `{split}/{class}/*.jpg`，实测通过 |
| R-09 | 混合形态数据集无法单布局导出 | **已解决** | 类别规范化（M1-07）提供 `kind_filter`：只保留一种标注形态，并自动清理因此变成空标注的图像。实测混合来源 8 类 → 检测 5 类 / 分类 3 类，两个方向都能正常导出 |
| R-14 | 同义类名识别的策略 | 已决策 | **只按归一化匹配（去标点+小写），不使用编辑距离等模糊匹配**。错误合并（把两个不同类合成一类）会污染类别语义且难以察觉，代价远大于漏合并。拼写差异交由人工在界面指定映射 |
| R-15 | VOC 坐标基准歧义 | 已决策 | Pascal VOC 原始规范是 **1-based 闭区间**，但 LabelImg 产出的是 **0-based**，两者混用会整体偏移 1 像素。默认**直接采用 XML 数值**（与主流转换工具一致），需要原始语义时传 `voc_one_based=True` |
| R-16 | 测试夹具稳定性 | 已修复 | 原用 `hash(uid)` 生成图像颜色，Python hash 每进程随机化导致偶发重复图被误删（flaky）。改为**结构性差异**（画位置不同的小方块）。注意：仅靠颜色差异不可靠——JPEG 量化会让相邻色号编码出完全相同的字节 |
| R-17 | 分桶函数的下溢处理 | 已修复 | 原实现把「小于首个边界」的值兜底计入**最后一桶**，导致每图目标数为 0 的图被算成"目标最多"的一档。改为归入首桶，并把首边界设为 0 以明确语义 |
| R-18 | 报告下载的越权风险 | 已加固 | `/api/files/download` 只允许下载存储目录内的文件，实测存储目录外的存在文件返回 403 |
| R-11 | 近似重复泄漏的残余 | 已知悉 | 分组划分后仍有 81 对跨子集近似重复，经查**全部为不同录制场次之间的相似帧**（同机位相隔数秒），非处理缺陷。同组跨子集为 0 |
| R-12 | 划分必须按「图像体积」而非「单元个数」分配 | 已修复 | 视频组单元 80 帧 vs 单图单元 1 帧，按个数分配会使 8:1:1 失真为 83.5:6.9:9.6。改为最大优先 + 最小缺口贪心后恢复为 80.0:10.0:10.0 |
| R-13 | `group_by` 需支持正则 | 已实现 | 数据集文件名格式各异（视频抽帧 vs 网图），`stem`/`parent` 不足以提取视频标识；新增 `regex:<pattern>` |
| R-10 | 血缘存储方式 | 已决策 | 用数据集自带的 `dataset_card.json` 记录血缘，不引入 sqlite。产物自描述，不会出现「数据库与实际文件不同步」 |
| R-19 | **ultralytics 对相对 `project` 路径的处理** | 已修复 | ultralytics 会把它自己的 `SETTINGS["runs_dir"]` 前缀拼到相对 `project` 上（实测得到 `backend\runs\detect\storage\runs\<job>`），导致 results.csv 写到别处、指标永远读不到。已在 `TrainingManager._apply_defaults` 强制把 `project` 解析为绝对路径 |
| R-20 | `data.yaml` 里 `path` 为相对路径时的解析基准 | 已知悉 | ultralytics 解析相对 `path` 的基准不是 yaml 所在目录（实测 `path: .` 被解析到 cwd）。本项目导出器写的 `path` 是**绝对路径**，不受影响；手工准备的数据集需写绝对 `path`（`tests/fixtures/make_tiny_det.py` 已按此生成） |
| R-21 | 接管运行中任务时的日志缺口 | 已知悉 | 服务重启后接管仍存活的训练进程，只能继续读 `results.csv`；重启前已经输出到 stdout 的内容不会补记（`train.log` 里只有该进程启动后写入的部分）。指标不受影响 |
| R-22 | pynvml 已安装但无 NVIDIA 设备 | 已处理 | `nvmlInit()` 抛 `NVMLError_LibraryNotFound`，资源监控降级为 `gpu.available=false` 且带 `error` 说明，CPU 面板照常。前端据此只显示 CPU 面板 |
| R-23 | numpy 数组的真值判断 | 已修复 | `extract_metrics` 里原写 `getattr(x, "p", []) or []`，numpy 数组会抛 `truth value of an array is ambiguous`。**只测 build_result 发现不了**（那条路径不碰 ultralytics 对象），是真实评估跑出来的。已统一改用 `_as_list()`，并补了用真 numpy 数组的回归测试 |
| R-24 | 指标缺失被当成 0 | 已修复 | `results_dict` 里的 NaN/非数值原先被写成 0.0，会把「没有这个指标」误读成「效果为 0」。改为**不写入该键**，逐类对比表中缺失项留空 |
| R-25 | 手工训练的任务在 API 里看不到 | 已修复 | 训练/评估管理器原只在启动时扫描目录，服务启动后用 CLI 新增的任务不可见。新增 `refresh()`，`list()` / `get()` 会自动补扫（`get` 未命中时重扫一次） |
| R-26 | 评估结果与模型库的一致性 | 已决策 | 模型卡片**不复制**指标与权重，只记录 `run_dir` / 权重路径与总体指标摘要；逐类指标与混淆矩阵始终从评估目录的 `eval_result.json` 读取，避免两处数据不一致 |
| R-27 | 导出格式的可用性判断 | 已决策 | 用 `importlib.util.find_spec` 探测依赖（**不真正 import**，避免把 onnx/openvino 拉进 API 进程）。不可用时在 `start()` 前就拒绝，原因写清缺哪个包 / 是否需要 GPU |
| R-28 | **ONNX 未安装，无法在本机真实验证** | 已知悉 | 本机只装了 torch，因此 ONNX / OpenVINO / TensorRT / TFLite / CoreML / RKNN 都无法真跑。**TorchScript 已真实验证**（导出 + 校验）；其余格式只验证了「能力探测与拒绝路径」。启用方式：`uv pip install onnx onnxslim`（+ 需要时 `onnxruntime` 做数值校验） |
| R-29 | 产物校验的口径 | 已决策 | 只校验**文件完整性**（存在性 / 大小 / sha256），不重新推理、不比较精度——那是评估模块（M3）的职责。目录型产物（如 OpenVINO）无法算单文件哈希，只比总大小 |
| R-30 | 导出产物是否随训练目录一起保留 | 已决策 | 默认把产物**复制**到 `storage/deploys/<deploy_id>/`，因为 ultralytics 会把导出文件写在权重旁边，而训练目录可能被清理。可用 `copy_artifacts=false` 关闭（测试覆盖了两种模式） |
| R-31 | 数据浏览器需要按筛选分页读取数据集 | 已解决 | 原 `/api/datasets/scan` 只返回前 20 张样本且无筛选，撑不起浏览器。新增 `POST /api/datasets/browse`：在内存 IR 上筛选（类别 / 划分 / 来源 / 标注形态 / 目标数 / 图像长边 / 路径子串）并按相对路径稳定分页；只读，不改原始数据 |
| R-32 | 代码分割后仍有偏大的 vendor chunk | 已优化 | ECharts 改为按需引入（M5-10）：echarts chunk **1,052 kB → 576 kB**（gzip 350→193 kB）。antd 核心 chunk 仍约 628 kB（gzip 204 kB），构建仍有 >500KB 警告——antd 本身即如此，进一步压缩需 `manualChunks` 或组件级按需，收益有限。实测强制把 antd 打进单一 chunk 反而会变成 1.1MB 且首屏全量加载，故**不采用** `manualChunks`，交给 Rollup 按页面自然分包 |
| R-33 | M5 页面的真实浏览器验证 | 已知悉 | 本会话桌面浏览器未连接，M5 页面只做了 `tsc + vite build` 与后端接口 / WebSocket 的真实联调；页面点击与图表渲染需在有浏览器的环境再复核一次 |
| R-34 | **ONNX 实际已安装**（更正 R-28） | 已更正 | 复核本机环境时实测 `onnx` 与 `onnxslim` **均已安装**（R-28 的「未安装」已过时），因此 ONNX **导出**现在就能跑。但 **`onnxruntime` 仍未安装**，ultralytics 的 ONNX **推理**依赖它 → 由 M6-07 顺带补上并真实验证 |
| R-35 | M6 实时验证的帧率上限 | 已决策 | 方案 A（浏览器抓帧 → WebSocket → 后端子进程推理 → 回传）在 CPU 上预期 **3~15 FPS**，够「肉眼验证」但不够流畅视频。前端做丢帧防堆积。要流畅需换前端推理/WebRTC，代价大，暂不做（见 `docs/m6-realtime-verify.md`） |
| R-36 | 推理日志格式与时间戳 | 已处理 | 训练/评估/导出日志里的时间戳使用本地时区，M6 的实时推理不落盘日志，避免高频帧写入磁盘 |
| R-37 | **M7 产出的是伪标签而非标注** | 已决策 | 模型预标注会**固化模型误差**（漏检=缺标注、错检=错标注），**必须人工复核后**才可作为训练数据。产出物与 dataset_card 都要标明「来自模型预测 + 权重 + conf」，不得暗示可跳过复核（见 M7 章节） |
| R-38 | 推理子进程的等待不依赖读线程唤醒 | 已修复 | Windows 上读线程的 `for line in stdout` **不保证在子进程退出时立刻返回**，导致「子进程崩溃」被误报成「超时」。改为等待循环主动 `poll()`（`InferSession._check_dead`）+ 有界 `wait(0.5s)`，崩溃能被立即感知 |
| R-39 | 推理会话不是线程安全的 | 已决策 | `InferSession` 的请求-响应配对靠共享字典，必须串行使用。API 层用 `session_lock()` 串行化；WS 用 `pending/processing` 状态机保证同一时刻只有一帧在推理（既保安全又实现丢帧） |
| R-40 | **ONNX 类名的说法不准确** | 已更正 | 原设计写「ONNX 不带类名，必须显式提供」——**不准确**。实测 `logix-fms-ahd/models/best_raw.onnx` 里通过 ultralytics 导出时写入了 metadata，`YOLO()` 能直接读出 5 个类名。已改为准确表述：ONNX 类名**不一定**有，有则自动读出；没有才必须显式提供。前端提示也从「必须填」改为中性说明 |
| R-41 | **显式类名没传到检测框上（真 bug）** | 已修复 | `describe_model` 报告了显式类名，但 `predict_image` 自己又 `_names_dict(model)` 读了一次模型类名，导致「加载时显示中文类名、框上仍是模型原类名」。**用户真实测试（对着摄像头举手机看到 `Phone`）才暴露出来**。已抽出共用的 `resolve_names()`，两处必须用同一份类名。会话快照也补上 `class_source` 字段 |
| R-42 | **stop() 可能返回中间态（训练/评估/导出共有的竞态）** | 已修复 | `test_eval` 偶发失败（约 1/6），暴露的是三个管理器共有的竞态：`_finalize` 要跑几秒（等进程 + 等日志刷完），若监控线程先进入且通过了 `finalized` 守卫，此时 `stop()` 里对 `_finalize` 的调用会**立即早返回**，于是拿回一个状态仍是 `stopping` 的任务。已加 `finalize_done` 事件：已有人在收尾时 `stop()` **等它真正跑完**再返回。三个管理器（train / eval / deploy）一并修，连跑 3 轮全量测试与 10 轮 `test_eval` 均无失败 |
| R-43 | **摄像头模式看不到框（真 bug）** | 已修复 | 实时验证页在摄像头模式下把 `video` 显示、`canvas` 设为 `display:none`，但**框正是画在这个隐藏 canvas 上**，因此画面上永远看不到框（用户实测「举着手机也没框」才暴露）。已改为 canvas 绝对定位叠在视频上（`pointerEvents:none`），并处理 `object-fit:contain` 黑边造成的坐标偏移（画布尺寸与位置对齐视频实际渲染区）。另把抓帧改为独立离屏 canvas，不再与显示画布混用 |
| R-44 | **类别清单只认英文逗号** | 已修复 | 前端原用 `split(',')`，用户打 `person;person1` 时**整串被当成一个类名**去覆盖某个下标（比报错更隐蔽）。已放宽为逗号 / 分号 / 竖线 / 顿号 / 换行等分隔符，并在界面写明「只改显示名字、不改变模型识别什么」及下标对应规则 |
| R-45 | **预标注任务存在内存里** | 已决策 | M7 的预标注任务（含未导出的 IR）保存在后端进程内存中，服务重启即丢失。理由：预标注是「跑一次 → 复核 → 导出」的一次性流程，把整批 IR 落盘会与数据集存储重复；重启后重跑即可。界面上标注了「仅本次后端进程内」 |
| R-46 | **预标注任务疑似串行推理**（自查） | 已更正 | 首版实现里 `_run_job` 逐张同步推理；实测 6 张 CPU 3.3s、5 张 ONNX 4.8s，批量几百张会较久。这是 CPU 推理的固有代价（与 M6 同源），不是缺陷。任务在后台线程跑、前端轮询，不阻塞 HTTP |
| R-47 | **`test_train` 存在环境相关的偶发失败（既有，与 M7 无关）** | 已知悉 | 现象：`test_api_layer` 偶尔在 `wait_terminal` 上**超时 25s**（失败运行总耗时 ~52s vs 正常 ~28s，差 ≈ 超时时间），随后 3 条断言失败（指标 1/2 行、增量取指标为空、已结束任务 stop 未 409），**无异常栈**——说明监控线程在 25s 内含异常地「没有收尾」。实测频率：单跑 `test_train.py` 约 1/8~1/12，**全量套件连跑时约 1/3**；把同一场景写成独立循环跑 40 次、以及给 `_finalize` 加异常探针后连跑 15 次，均**不复现**，故强依赖测试文件上下文 / 机器负载，而非单段逻辑。涉事代码是 `core/train/manager.py`（**M7 完全未改动**）。两个怀疑点：①`_finalize` 先置 `rt.finalized=True` 再干活，若中途抛错则再次进入只会 `finalize_done.wait()` 而永不落终态（R-42 的 `finalize_done` 会让这种 wedge「静默等待」）；②监控线程被 OS/AV 抢占长时间未 tick。**同类偶发也见于 `test_eval`**（一次性连跑 5 个文件时出现 140/2，单独跑仍 147 全过），指向环境/负载而非某个模块。**当前结论：以单次全量运行为准，但该文件并非稳定全绿。** 根治方向：收尾体加 `try/finally` 兜底落终态 + 给监控循环加看门狗；测试侧延长超时并打印超时诊断 |

---

## 验证记录

| 日期 | 项目 | 结果 |
|---|---|---|
| 2026-09-22 | core 解析 `train_dms_model/data/archive` | 9884 图 / 23389 标注 / 5 类 / 0 警告，划分 5957:2389:1538 |
| 2026-09-22 | CLI `scan` 命令 | 通过 |
| 2026-09-22 | 后端接口 `/api/system/*`、`/api/datasets/*`、`/api/files/image` | 全部 200 |
| 2026-09-22 | 前端 `npm run build`（tsc --noEmit + vite build） | 通过 |
| 2026-09-22 | 后端托管生产前端（`GET /`） | 200，SPA index.html 正常返回 |
| 2026-09-22 | 训练环境探测 `/api/system/env` | ultralytics 8.4.160 / torch 2.8.0+cpu，`training_python` 指向 `backend/.venv` |
| 2026-09-22 | SPA 兜底不再吞掉 `/api/*` 的 404 | `/api/nope` 正确返回 404 |
| 2026-09-22 | 接入层测试 `backend/tests/test_ingest.py` | **全部通过**（40+ 断言） |
| 2026-09-22 | OpenLABEL 适配器（DMD 夹具，60 帧） | 3 类 / 60 图像级标注 / 0 警告；区间边界逐帧核对通过 |
| 2026-09-22 | OpenLABEL 层级选择与 object 并入 | `gaze_on_road` 层级、`level=None` 全层级、`include_objects` 均正确 |
| 2026-09-22 | 原 DMS 数据集回归（YOLO 适配器） | 9884 图 / 23389 框，标注形态 `bbox`，无回归 |
| 2026-09-22 | 划分 + 导出测试 `backend/tests/test_pipeline.py` | **全部通过**（89 断言） |
| 2026-09-22 | 真实数据全规模导出（DMS 9884 图） | 23389 框，hardlink 模式。当时的「11 个越界框」经复核为**浮点噪声**（坐标误差 ~1e-6），非真实越界；导出判定已加入 1e-3 容差 |
| 2026-09-22 | 类别下标保持性 | 导出前后逐标签比对 5957 个文件：类别倒错位 0，坐标最大误差 1e-6（640px 图上 0.0006px） |
| 2026-09-22 | 多来源合并（YOLO + OpenLABEL） | 9884 + 60 = 9944 图 / 8 类，正确识别为 `mixed` |
| 2026-09-22 | 同一来源重复导入 | 正确去重（9884 而非 19768），并有警告 |
| 2026-09-22 | 混合形态导出保护 | HTTP 400，提示无法用单一布局导出（不产出坏数据） |
| 2026-09-22 | 接口 `/api/datasets/split`、`/export`、`/versions` | 全部 200 |
| 2026-09-23 | 清洗测试 `backend/tests/test_clean.py` | **全部通过**（58 断言） |
| 2026-09-23 | 近似重复加速算法 vs 暴力算法 | 3 组随机数据集结果完全一致（不丢对）；2 万图 < 1s |
| 2026-09-23 | 真实数据集清洗（9884 图，全量） | 检出 1160 条问题（335 error），耗时 38.8s |
| 2026-09-23 | 清洗前后泄漏对比（真实数据） | 跨子集完全重复 115→**0**，同子集重复 41→**0** |
| 2026-09-23 | 分组划分防泄漏（regex 提取 114 个视频组） | 跨子集分组数 **0**；近似重复泄漏 220→81（残余全为跨场次固有相似） |
| 2026-09-23 | 划分比例修正（按体积分配） | 83.5:6.9:9.6 → **80.0:10.0:10.0** |
| 2026-09-23 | 接口 `/api/datasets/clean`、`/clean/rules` | 200，全量检查 15.1s |
| 2026-09-23 | 类别规范化测试 `backend/tests/test_taxonomy.py` | **全部通过**（69 断言） |
| 2026-09-23 | **R-09 解决**：混合来源 + `kind_filter` | 8 类 → 检测 5 类（移除 60 条图像级标注、60 张空图）／分类 3 类（移除 23389 条检测框、9884 张图），均成功导出 |
| 2026-09-23 | 非法 `kind_filter` 拒绝 | HTTP 400，明确提示合法取值 |
| 2026-09-23 | 真实数据类别检查 | 5 类，无同义类名、无非法类名 |
| 2026-09-23 | 适配器测试 `backend/tests/test_adapters.py` | **全部通过**（55 断言） |
| 2026-09-23 | 三种格式端到端往返（COCO/VOC/LabelMe → YOLO → 再接入） | 图像数、标注数、类别分布全部一致 |
| 2026-09-23 | 格式探测不串台 | 各适配器不误判其它格式；原 DMS 数据集与 DMD 夹具仍判定为 yolo / openlabel |
| 2026-09-23 | 三格式经 API 合并 | 12 图（5+4+3）、3 个分组，划分报告含实际比例偏差 |
| 2026-09-23 | 清洗测试稳定性 | 连续 6 次运行全部通过（修复 flaky 前为随机失败） |
| 2026-09-23 | 统计分析测试 `backend/tests/test_analytics.py` | **全部通过**（84 断言） |
| 2026-09-23 | 真实数据集质量报告（9884 图） | 检出**子集类别构成偏移**：test 的 Cigarette 32%（整体 17%）、Open Eye 36%（整体 48%）；不均衡比 6.97× |
| 2026-09-23 | HTML 报告生成与下载 | 89.9 KB 单文件，章节完整、缩略图 base64 内嵌、可下载 |
| 2026-09-23 | 报告下载越权保护 | 存储目录外的存在文件返回 403 |
| 2026-09-23 | 全量测试 | **404 断言全部通过**（6 个测试文件） |
| 2026-09-23 | 训练模块测试 `backend/tests/test_train.py` | **全部通过**（142 断言）；覆盖后端抽象、指标解析、状态机、重启接管、资源降级、产物分组、API 与 WS 桥接、日志落盘 |
| 2026-09-23 | 真实 WebSocket 联调（对运行中的后端） | 订阅 `smoke_yolo11n` 收到 `snapshot`（含指标/日志/资源）；不存在的任务以关闭码 **4404** 拒绝 |
| 2026-09-23 | **真实训练（非模拟）**：yolo11n.yaml 从零，CPU，imgsz=32，1 epoch | `finished`，退出码 0；`results.csv` 解析出 1 轮，train_loss=0.08369；产出 12 张过程图 + best/last.pt |
| 2026-09-23 | **真实停止 + 断点续训**：epochs=3，第 1 轮后停止再续训 | 停止后 `stopped`（rc=15，last.pt 已写）；续训后 `finished`（rc=0），epochs=[1,2,3]，progress=1.0 |
| 2026-09-23 | 真实训练产物经 API 读取（新进程） | 恢复 61 行日志、1 轮指标、12 张过程图；分组 val_pred/val_labels/train_batch/labels/curve/matrix 正确 |
| 2026-09-23 | 修复 R-19（相对 project 路径） | 修复前 ultralytics 把结果写到 `backend\runs\detect\storage\runs\...`；修复后 `save_dir` 与训练目录一致 |
| 2026-09-23 | API 进程依赖隔离 | 子进程验证：构建命令与训练参数**不加载** ultralytics / torch |
| 2026-09-23 | 全量测试（含训练模块） | **546 断言全部通过**（7 个测试文件） |
| 2026-09-23 | 评估模块测试 `backend/tests/test_eval.py` | **全部通过**（147 断言）；覆盖结果归一化、生命周期、重启接管、报告、模型注册、多模型对比、API |
| 2026-09-23 | **真实评估（非模拟）**：`smoke_yolo11n/weights/best.pt` 在 tiny_det 的 test 划分上 val | `finished`，rc=0；抽出总体 6 项指标、逐类指标、3×3 混淆矩阵（含 background）、8 张过程图并正确分组 |
| 2026-09-23 | **真实端到端（对运行中的后端）**：注册 → 评估 → 卡片 → 对比 → 报告 | 2 个真实模型注册成功；test 划分评估 rc=0 并自动挂到卡片；对比返回 2 行、同划分、指标按序；HTML 报告 436 KB |
| 2026-09-23 | 修复 R-23 | 真实评估暴露 numpy 数组真值判断缺陷；补真数组回归测试后通过 |
| 2026-09-23 | 修复 R-24 / R-25 | 缺失指标不再写成 0；API 能发现服务启动后用 CLI 新增的任务 |
| 2026-09-23 | 全量测试（含评估模块） | **693 断言全部通过**（8 个测试文件） |
| 2026-09-23 | CLI `eval` 子命令真实评估 | `finished`，退出码 0；输出总体/逐类指标与混淆矩阵，8 张过程图 |
| 2026-09-23 | 修复**事件顺序缺陷**：`finished` 早于上层联动 | 原先先发 `finished` 再挂模型卡片，客户端收到事件后立刻取卡片会读到旧数据。改为「先联动、再发事件」，并连续 3 次运行测试确认不再 flaky |
| 2026-09-23 | 修复**测试污染真实 storage** | 自动评估/注册联动上线后，`test_train` 的训练完成回调会写真实 `storage/models` 与 `storage/evals`。改为注入隔离的 registry 并显式关闭自动评估；评估报告改写到评估目录内，避免写全局 reports |
| 2026-09-23 | 部署导出测试 `backend/tests/test_deploy.py` | **全部通过**（103 断言）；覆盖格式探测、export_kwargs、产物哈希与校验、成功/部分成功/全部失败/停止/接管、API、模型卡片联动 |
| 2026-09-23 | **真实导出（非模拟）**：TorchScript，`smoke_e3/weights/best.pt`，imgsz=32 | `finished`，rc=0；产物 10.27 MB 复制到导出目录；`verify` 通过（1 个产物，0 问题） |
| 2026-09-23 | **真实端到端（对运行中的后端）**：模型库触发导出 | `/api/deploy/formats` 逐一报告可用性；`model_id=smoke_e3` 导出 torchscript 成功；产物挂到模型卡片，摘要显示 `deploy_formats=torchscript` |
| 2026-09-23 | 能力探测与拒绝路径 | 未装 `onnx` / 无 CUDA 的格式均在启动前被拒绝并说明原因（不启动注定失败的子进程） |
| 2026-09-23 | 全量测试（含部署导出） | **796 断言全部通过**（9 个测试文件） |
| 2026-09-24 | 新增数据浏览器接口 `POST /api/datasets/browse` | 在 `tiny_det`（14 图）上实测：总数 14、分页 3 张正确；`split=test` → 1；`category=square` → 1；`min_objects>=1` → 14；返回带 `source_id` 与受限图片 url |
| 2026-09-24 | 数据浏览器接口测试（并入 `test_ingest.py`） | 新增 14 断言，`test_ingest.py` 49 → **63**，全部通过 |
| 2026-09-24 | 全量测试 | **810 断言全部通过**（9 个测试文件，`test_ingest` 63 / `test_train` 142 / `test_eval` 147 / `test_deploy` 103） |
| 2026-09-24 | 前端 `npm run build`（tsc --noEmit + vite build） | 通过；路由级 lazy 生效，各页面拆为独立 chunk（Dashboard / TrainJobs / TrainDetail / ModelDetail / ModelCompare / DatasetBrowser 等） |
| 2026-09-24 | **真实接口联调（对运行中的后端 8010）** | `/api/train/jobs`（2 个真实任务）、`/api/models`（2 个真实模型）、`/api/deploy/formats`（仅 torchscript 可用）、`/api/train/jobs/smoke_e3`（3 轮指标 / 71 行日志 / CPU 可用 GPU 不可用）、`/api/train/jobs/smoke_e3/artifacts`（13 图 / 6 组）、`/api/models/smoke_e3/result`（逐类 2 类 + 混淆矩阵）、`/api/models/compare`（2 行、同 test 划分）、`/api/eval/.../artifacts`（8 图）、`/api/deploy/.../result` 与 `/verify`（torchscript 10.3MB，校验通过）全部 200 |
| 2026-09-24 | **真实 WebSocket 联调** | 订阅 `smoke_e3` 收到首帧 `snapshot`；不存在的任务以关闭码 **4404** 拒绝（与训练模块约定一致） |
| 2026-09-24 | ECharts 按需引入（M5-10） | 构建产物 echarts chunk **1,052.15 kB → 576.47 kB**（gzip 349.71→192.76 kB）；审计全站仅用 line / bar / heatmap 三种 series，均已注册；`echarts-for-react/lib/core` 确认不引入全量 echarts |
| 2026-09-24 | 前端构建产物总量 | dist JS 合计约 1,863 KB；入口 chunk 629 KB（antd 核心 + 布局），各页面独立 chunk，懒加载 |
| 2026-09-24 | 生产前端托管回归 | `/`、`/train/:id`、`/models/compare`、`/data/browser` 深链均返回 SPA index（200），入口资源 200，`/api/nope` 仍 404 |
| 2026-09-28 | 配置管理可覆盖性（M0-04 复核） | 设 `YOLO_STUDIO_RUNS=D:\tmp\runs_override` 后新进程读到的 `runs_dir` 即为覆盖值，确认环境变量生效 |
| 2026-09-28 | 新增 `GET /api/system/config` 与设置页（M5-11） | 接口 200：10 个路径 + 11 项可选依赖 + 7 种导出格式；前端经 Vite 代理 `/api/system/config` 200；后端重启后旧任务/模型照常加载 |
| 2026-09-30 | M6 测试 `backend/tests/test_infer.py` | **全部通过**（89 断言）；覆盖格式探测、路径推断、参数校验、结果归一化容错、会话生命周期、崩溃/超时感知、worker 行协议、API、背压丢帧 |
| 2026-09-30 | **真实推理（非模拟）**：`smoke_e3/weights/best.pt`，imgsz=64，CPU | 子进程加载成功（2 类 square/circle）；路径推理与 base64 帧推理均返回结构化结果；0 框属正常（12 图冒烟模型 mAP=0） |
| 2026-09-30 | **真实 ONNX 导出 + 推理（首次跑通）** | M4 导出 ONNX 成功：10.0 MB，`verify` 通过；用 `onnxruntime==1.19.2` 对导出的 ONNX 真实推理成功（显式给类名）。**这是本项目第一次真实验证 ONNX 链路**（此前 R-28 只有拒绝路径） |
| 2026-09-30 | **真实 API 联调（对运行中的后端 8011）** | `/api/infer/formats` 200（7 格式，pt/onnx/torchscript 可用）；`/weights` 200（4 条：2 模型库 + 2 导出产物）；`/models` 200 加载；`/image` 200（path 与 base64 两条路径）；未加载模型推理正确 400 |
| 2026-09-30 | **真实 WebSocket 联调** | `ws://…/api/infer/ws`：`ready` → `loaded` → 连发 8 帧全部返回 `result` → `closed`，退出干净 |
| 2026-09-30 | CLI `infer` 真实推理 | `--list-formats` 列出 7 格式与可用性；对 2 张图推理，输出任务/类名/耗时 |
| 2026-09-30 | 全量测试（含 M6） | **899 断言全部通过**（10 个测试文件：810 → 899，M6 新增 89） |
| 2026-09-30 | **真实浏览器验证 M6-06（用户操作）** | 页面 `/verify` 真实打开：摄像头出画面（`getUserMedia` 成功）、逐帧推理持续进行（215 帧 / 6.3 FPS / 0 丢弃）。真实权重 `logix-fms-ahd/models/best_raw.onnx`（5 类 DMS 模型）在图片模式下**画出 `Phone` 框**，叠加链路正常 |
| 2026-09-30 | 修复 R-40 / R-41（真实测试暴露） | ①ONNX 类名说法不准确（实测该 ONNX 自带 names，能直接读出）；②**显式类名没传到框上**——`predict_image` 重复读模型 names。已抽出共用 `resolve_names()`。修复后用真实 ONNX 复验：显式填中文类名后，**框上确实显示中文**而非 `Phone` |
| 2026-09-30 | 全量测试（含修复） | **907 断言全部通过**（M6 97）；前端 `npm run build` 通过 |
| 2026-09-30 | 修复 **R-42**：`stop()` 可能返回中间态。`test_eval` 偶发失败（约 1/6）暴露了 train/eval/deploy 三个管理器共有的竞态——监控线程与 `stop()` 同时进 `_finalize` 时，后者会早返回而拿到仍为 `stopping` 的任务。加 `finalize_done` 事件，让 `stop()` 等收尾真正跑完。**连跑 3 轮全量测试 + 10 轮 `test_eval`，零失败**（修复前约 1/6 失败） |
| 2026-09-30 | 新增 M7 测试 `backend/tests/test_prelabel.py` | **全部通过（97 断言）**；覆盖 list_images（递归/单文件/缺目录）、build_bundle（图像数/uid/尺寸/相对路径）、annotate_bundle（正常/`boxes_total` 累加/逐类计数/meta 血缘）、单个图业务失败与子进程崩溃均记入 failures 且不中断整批、空目录、分类分支（kind=image、bbox=None）、血缘写入、导出卡片、API 路由与错误路径。用假 worker 脚本（不依赖 torch），崩溃用例走真实常驻子进程 |
| 2026-09-30 | **修复预标注历史统计 bug 的回归** | `boxes_total` 未累加曾导致「报告显示 0 框、实际 21 个」。本轮补测试时同时修正 `annotate_bundle`：**worker 回 `ok=false`（业务错误）不再当成「空标注」**，改记入 `report.failures`，避免把失败混进空图（与 R-37「漏检不可掩盖」同理） |
| 2026-09-30 | **真实 API 联调（M7-02，TestClient + 真实权重）** | 用 `smoke_e3/weights/best.pt`（2 类）对 `tiny_det/images`（14 图）中 6 张预标注：任务 `finished`，进度 6/6，`images_total=6 / 失败 0 / 类别 ['square','circle']`（mAP=0 的冒烟模型，0 框属正常）；samples 返回 6 张预览；导出 200，目录名 `prelabel_verify_api`（前缀生效），`dataset_card.json` 的 `prelabel` 段含权重/conf/classes/disclaimer |
| 2026-09-30 | **真实 CLI 联调（M7 CLI + 真实 ONNX，证明有框）** | `cli.main prelabel` 用 DMS ONNX（5 类，`best_raw.onnx`）对 5 张图：**15 个框**（Phone 12 / Cigarette 3），导出 train3/val1/test1、`boxes_exported=15`；**磁盘上 labels 逐文件 6+1+4+3+1=15 行**，与报告一致；`dataset_card.json` 的 `prelabel` 段完整（权重/conf/iou/classes/disclaimer） |
| 2026-09-30 | 前端 `npm run build`（M7 向导页） | 通过；新增 `pages/PrelabelWizard.tsx`（三步向导，chunk 38.0 kB）、`/prelabel` 路由与「预标注」菜单 |
| 2026-09-30 | M7 API 错误路径 | 不存在的任务 404、图片目录不存在 404、权重不存在 404（启动前拒绝，不跑空任务） |

---

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-22 | 创建项目骨架与进度表；确定技术栈 |
| 2026-09-22 | 完成 M0 全部任务；完成 M1-01（IR）、M1-02（YOLO 适配器）、M5-01/02（布局与总览） |
| 2026-09-22 | 建立后端 venv `backend/.venv`（Python 3.9.25） |
| 2026-09-22 | 训练依赖（ultralytics 8.4.160 + torch 2.8.0）纳入 `requirements.txt` |
| 2026-09-22 | API 端口由 8000 改为 8010（避开本机 LLM 网关）；修复 SPA 兜底吞掉 `/api/*` 404 的问题 |
| 2026-09-22 | 完成 **M1-06**：OpenLABEL/VCD 适配器（DMD）；IR 扩展 `kind` 字段以支持图像级分类标注；新增测试夹具与接入层测试 |
| 2026-09-22 | 确认 R-08：DMD 为分类任务（无 bbox），M1-10 导出需支持分类目录布局 |
| 2026-09-22 | 完成 **M1-09 划分**（分组防泄漏 / 分层 / 可复现）与 **M1-10 导出**（检测+分类双布局） |
| 2026-09-22 | 完成 **M1-11 血缘**（dataset_card.json + 版本列表页）与 **M5-03 导入向导**（扫描→合成清单→划分→导出） |
| 2026-09-22 | 补齐**多来源合并**入口（CLI/API/前端），修复 uid 不含根路径导致同名目录撞车的隐患 |
| 2026-09-22 | 修复 `category_names()` 排序：改为优先采用数据源声明顺序，避免导出后类别下标重排 |
| 2026-09-22 | 修复划分模块缺陷：最小样本量修正改为**全局**执行（原按分层执行会使 val 占比被分层数放大） |
| 2026-09-23 | 完成 **M1-08 清洗引擎**：16 条规则（图像/重复/标签/泄漏/类别），dry-run + apply，接入 CLI/API/前端 |
| 2026-09-23 | IR 增加标注索引（`annotation_index`），消除 `annotations_of` 的 O(n) 全表扫描 |
| 2026-09-23 | 近似重复检测改为**多重索引**加速（鸽巢原理分 7 段建倒排索引），完成度：结果与暴力法一致 |
| 2026-09-23 | YOLO 适配器新增孤儿标签反向扫描；新增 `group_by=regex:<正则>` |
| 2026-09-23 | **修复划分比例失真**：分配改为按图像体积的最大优先贪心；并新增实际比例报告与偏差告警 |
| 2026-09-23 | 导出判定加入 1e-3 容差，避免浮点噪声被误报为「越界框」 |
| 2026-09-23 | 完成 **M1-07 类别规范化**：同义类名建议、keep/drop/重排/sanitize、**按标注形态过滤**；接入 CLI/API/前端 |
| 2026-09-23 | **解决 R-09**：混合形态数据集现在可通过 `kind_filter` 过滤后正常导出，不再只能报错 |
| 2026-09-23 | 修复类别表残留缺陷：被丢弃的类别仍留在类别表中，会导致导出生成 0 实例的幻影类别与空目录 |
| 2026-09-23 | 完成 **M1-03/04/05**：COCO / Pascal VOC / LabelMe 三个适配器；支持 `images_dir`、`voc_one_based` 参数 |
| 2026-09-23 | 修复测试夹具的 flaky 问题（`hash()` 导致非确定性），改用结构性差异保证图片互不相同 |
| 2026-09-23 | 完成 **M1-12 统计分析**：类别/尺寸/密度分布、子集构成偏移检测、单文件 HTML 报告；新增质量报告页（ECharts） |
| 2026-09-23 | **M1 数据模块 12/12 全部完成** |
| 2026-09-23 | 修复分桶下溢缺陷（小于首边界的值被计入末桶）；加固报告下载的越权保护 |
| 2026-09-23 | 完成 **M2 训练模块 7/7**：`TrainerBackend` 抽象 + ultralytics 实现、subprocess 隔离的状态机（start/stop/resume）、results.csv 指标解析、WebSocket 增量推送、CPU/GPU 资源监控与降级、过程图像浏览、日志落盘与流式推送 |
| 2026-09-23 | 新增训练接口 `/api/train/*`（jobs / metrics / logs / resources / artifacts / image / ws）与 CLI `train` 子命令（含 `--dry-run`） |
| 2026-09-23 | 修复 **R-19**：ultralytics 对相对 `project` 会拼上自身 runs_dir，导致指标读不到；训练目录强制绝对路径 |
| 2026-09-23 | 修正 M5 进度计数：`3/8` → **`4/8`**（M5-01/02/03/06 已完成） |
| 2026-09-23 | 新增真实训练冒烟数据集生成器 `tests/fixtures/make_tiny_det.py`（8 train + 4 val + 2 test，2 类） |
| 2026-09-23 | 完成 **M3 评估与模型库 4/4**：独立子进程评估（val/test）、逐类指标 + 混淆矩阵 + 单文件 HTML 报告、模型库 `model_card.json`（血缘 + 权重 + 评估索引）、多模型对比（同划分校验） |
| 2026-09-23 | 新增接口 `/api/eval/*`（jobs/result/logs/artifacts/image/report/ws）与 `/api/models/*`（register/list/compare/detail/result/eval） |
| 2026-09-23 | 抽出公共 `core/process.py`（`pid_alive` / `kill_tree`），训练与评估调度共用，避免各写一份 |
| 2026-09-23 | 修复 R-23（numpy 数组真值判断，真实评估才发现）、R-24（缺失指标写成 0）、R-25（CLI 新任务在 API 不可见） |
| 2026-09-23 | 修复**训练终态竞态**：原先先置 `finished` 再解析最终指标，外部可能读到残缺指标；改为先算指标与日志、最后置终态 |
| 2026-09-23 | 修复**联动与事件顺序**：先让上层注册模型/挂评估结果，再推送 `finished`，避免客户端拿到事件后读到旧卡片 |
| 2026-09-23 | 新增 CLI `eval` 子命令（评估权重并打印总体/逐类指标与混淆矩阵） |
| 2026-09-23 | 修复测试隔离：训练/评估用例不再写入真实 `storage/`（注入隔离 registry、显式关闭自动评估）；评估报告改写到评估目录内 |
| 2026-09-23 | 完成 **M4 部署导出 3/3**：多格式导出（子进程隔离）、格式能力探测与明确拒绝、产物复制 + 大小/sha256 基线、`verify` 完整性校验、产物挂到模型卡片 |
| 2026-09-23 | 新增接口 `/api/deploy/*`（formats/jobs/result/verify/logs/download/ws）与 CLI `export-model`（含 `--list-formats`） |
| 2026-09-23 | 新增配置 `YOLO_STUDIO_DEPLOYS`；`ModelCard` 增加 `deploys` 字段 |
| 2026-09-24 | 完成 **M5-07 训练列表 / 新建页**：`TrainJobs.tsx`（状态/进度/最优指标，运行中自动轮询，停止/续训/注册）、`TrainNew.tsx`（环境探测 + 数据集/权重/超参表单，无 CUDA 提示） |
| 2026-09-24 | 完成 **M5-05 训练详情页**：`TrainDetail.tsx` + `useJobSocket`；WebSocket 增量驱动实时曲线（指标可切换）、CPU/GPU 资源面板、网页终端日志、过程图像与参数/产物 |
| 2026-09-24 | 完成 **M5-08 模型库与对比页**：`ModelLibrary.tsx`（多选对比）、`ModelDetail.tsx`（概览/评估/导出部署，含发起评估、HTML 报告、格式可用性、产物下载与 verify）、`ModelCompare.tsx`（总体指标高亮 + 逐类 AP50-95 + 柱状图） |
| 2026-09-24 | 完成 **M5-04 数据浏览器**：`DatasetBrowser.tsx` + 后端 `POST /api/datasets/browse`（分页与多条件筛选） |
| 2026-09-24 | 完成 **M5-09 路由级代码分割**：`App.tsx` 全部页面 `React.lazy`，`MainLayout` 内 `Suspense` 兜底 |
| 2026-09-24 | 前端补齐 API 封装与类型：新增 train/eval/models/deploy/browse 方法与 `wsUrl` 辅助；`vite.config.ts` 代理开启 `ws: true`（否则开发环境 WS 无法经代理） |
| 2026-09-24 | **M5 前端界面全部完成（9/9）** |
| 2026-09-24 | 完成 **M5-10 ECharts 按需引入**：新增 `components/echarts.tsx`（只注册 line/bar/heatmap + grid/tooltip/legend/visualMap + Canvas），四个图表组件改用它；echarts chunk 减半 |
| 2026-09-24 | 放弃 `manualChunks` 强制分包（实测把 antd 打成单 chunk 会变 1.1MB 且首屏全量加载，得不偿失），改为依赖 Rollup 按页面自然分包 |
| 2026-09-24 | **M5 前端界面 10/10**（新增产物优化任务） |
| 2026-09-28 | 新增只读配置接口 `GET /api/system/config`（路径 / 可选依赖 / 导出能力 / 读取白名单） |
| 2026-09-28 | 完成 **M5-11 设置页**：把 `/settings` 从占位改为真页面；删除已无用的 `Placeholder.tsx`；澄清 M0-04（后端配置管理）本就已完成 |
| 2026-09-28 | **M5 前端界面 11/11** |
| 2026-09-29 | 从总览页移除**过期路线图卡片**（M0~M5 硬编码进度与 `PROGRESS.md` 不一致）与页头 `骨架版 v0.1` 标签；修正「已接入的数据格式」下的过期注释（原称仅实现 YOLO）。**进度仍以 `PROGRESS.md` 为唯一事实来源**，界面不再展示以免制造第二事实来源 |
| 2026-09-29 | 导入向导「类别体系」面板新增**任意自定义映射**编辑器（`ExportStep.tsx`）：可把任意类名改名为任意目标（含类别清单外的新名称，用 `mode=tags` 输入），多行指向同一目标即合并为一类。自定义映射**优先级高于系统建议**（在 `buildMapping` 中最后应用，避免被建议覆盖）。补齐 R-14 承诺的「拼写差异交由人工在界面指定映射」——此前该入口仅存在于 CLI/API，界面上只能改系统检测出的同义组。后端 `mapping` 本就支持任意映射，无需改动；实测系统建议为空时，3 类 → 2 类（多对一 + 改名 + 全新目标名）全部生效 |
| 2026-09-30 | 新增 **M6 实时验证** 设计文档 `docs/m6-realtime-verify.md` 并在 `PROGRESS.md` 建立 M6（8 项）/ M7（预标注，3 项）任务清单。**M6 定位：无真值、肉眼看识别情况，不产出数据集，不落盘**（与 M3 评估明确区分）。采用方案 A（浏览器抓帧 → WebSocket → 常驻推理子进程 → 回传叠加）；权重格式走能力探测；`.pt` 必支持，ONNX 待装 `onnxruntime` |
| 2026-09-30 | **更正 R-28**：实测本机 `onnx` 与 `onnxslim` **已安装**（此前记为未装），ONNX 导出链路现已可用；`onnxruntime` 仍未装，M6-07 一并补上并真实验证。另修正 HANDOFF 里「未安装的可选依赖」清单 |
| 2026-09-30 | 完成 **M6 实时验证后端（7/8）**：新增 `core/infer/`（格式探测、参数契约、常驻推理子进程 + JSON Lines 协议、结果归一化）与 **API**（`/api/infer/formats|weights|session|models|close|image` + `/ws`）。方案 A：浏览器抓帧 → WebSocket → 后端子进程推理 → 回传叠加；**API 进程不加载 torch**（沿用 M2/M3/M4 隔离原则）。已装 `onnxruntime` 并**首次真实验证 ONNX 导出 + 推理**。待做 M6-06 前端页面 |
| 2026-09-30 | 修复 **R-38**：推理子进程崩溃被误报为「超时」——Windows 上读线程不保证在子进程退出时立即返回，改由等待循环主动 `poll()` + 有界等待 |
| 2026-09-30 | 新增 CLI `infer` 子命令（`--list-formats` / 多图 / `--json` / `--save`）；`requirements.txt` 补记可选依赖 `onnxruntime` 与测试用 `httpx` |
| 2026-09-30 | 完成 **M6-06 前端验证页**：新增 `pages/VerifyCenter.tsx`（权重选择 / 参数 / 摄像头·图片·视频切换 / canvas 叠加 / 实时 FPS 与丢帧统计）、`/verify` 路由与侧边栏「实时验证」菜单；`api/client.ts` 与 `types.ts` 补 M6 接口与类型。**已通过真实浏览器验证**（摄像头出画面、叠加生效） |
| 2026-09-30 | 修复 **R-41**：显式类名未传到检测框（`predict_image` 重复读模型 names）→ 抽出共用 `resolve_names()`；顺带更正 R-40 对 ONNX 类名的表述。**由用户真实操作暴露**，非测试可得 |
| 2026-09-30 | `InferSession.snapshot()` 补 `class_source`（request / model），前端显示类名来源，避免「不知道类名是哪来的」 |
| 2026-09-30 | 修复 **R-42**：训练/评估/导出三个管理器共有的 `stop()` 竞态（返回中间态）。由 `test_eval` 偶发失败暴露 |
| 2026-09-30 | 修复 **R-43 / R-44**（前端，均由用户实测暴露）：①摄像头模式把框画在隐藏的 canvas 上，导致永远看不到框；②类别清单只认英文逗号，用户用分号时整串被当成一个类名。已改为叠加显示 + 多分隔符 + 对齐 contain 黑边 |
| 2026-09-30 | 完成 **M7-01 批量推理 → IR（伪标注）**：`core/prelabel/` 新增 `run_prelabel` / `attach_lineage` / `ensure_prelabel_prefix` / `export_name`；`annotate_bundle` 修正业务错误被当成空标注的问题 |
| 2026-09-30 | 完成 **M7-03 血缘**：`core/export` 的 `dataset_card.json` 固定写入 `prelabel` 段（权重 / conf / iou / 模型类别 / 生成时间 + 免责说明）；导出目录名经 `ensure_prelabel_prefix` 强制带 `prelabel` 前缀 |
| 2026-09-30 | 完成 **M7-02 API**：新增 `app/services_prelabel.py`（后台线程任务 + 独立推理会话 + 复用 taxonomy/clean/split/export）与 `app/api/routes/prelabel.py`（formats/weights/jobs/samples/export/delete）；新增 `PrelabelJob` 等 schema |
| 2026-09-30 | 完成 **M7 前端向导**：`pages/PrelabelWizard.tsx`（权重+目录 → 预览复核 → 生成数据集）+ `/prelabel` 路由与菜单；`api/client.ts`、`types.ts` 补齐 M7 接口与类型。页面与首页注明「伪标签必须人工复核」 |
| 2026-09-30 | 完成 **M7 CLI**：`cli.main prelabel`（`--list-formats` / `--limit` / `--out` 顺手导出 / `--json`），与 `infer` / `export-model` 风格一致 |
| 2026-09-30 | **M7 预标注 3/3 全部完成**；新增测试 97 断言。全量应为 **1004 断言**（11 个测试文件）；实测 10 个文件（含 `test_prelabel` 97）**稳定全绿**，唯一不稳的是 `test_train` 的既有偶发（R-47，M7 未改动其代码） |
| 2026-09-30 | 复核基线：`test_train.py` 单跑 8 次中 1 次、12 次中 1 次失败；全量套件连跑时约 1/3 失败（均同 3 条断言、耗时 +≈25s，见 R-47）。其余 10 个测试文件连同 `test_prelabel` **零失败**。**更正**：此前「907 全绿」只代表单次运行，`test_train` 本身存在环境相关偶发 |
