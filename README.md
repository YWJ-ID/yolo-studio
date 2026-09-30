# YOLO Studio

可视化 YOLO 训练全流程的一体化工作台：**多源数据合并 → 数据清洗与规范化 → 数据集划分 → 训练调度 → 实时可视化**。

> 定位：把「一堆格式各异的脏数据」变成「干净、可复现、可训练的数据集」，并让训练过程**看得见**。

---

## 为什么需要它

现有 `train_dms_model` 是命令行脚本，存在三个缺口：

1. 只支持 Roboflow YOLOv8 一种数据格式，无法合并 COCO / VOC / LabelMe / OpenLABEL(DMD) 等来源；
2. 没有数据清洗与跨集合泄漏检测，脏数据会直接污染指标；
3. 训练过程只有 ultralytics 生成的静态图，无法实时观察。

本项目把这三件事做成一条可视化流水线。

---

## 架构

```
core（纯 Python 库，CLI 可直接用）
  ↑
api（FastAPI + WebSocket）
  ↑
web（React + Ant Design + ECharts）
```

**硬约束**：数据管道逻辑全部在 `core`，绝不写进 `api`/`web`。保证 CLI 与界面行为永远一致。

目录结构：

```
yolo-studio/
├── PROGRESS.md              # 任务进度追踪（唯一事实来源）
├── docs/                    # 设计文档
├── backend/
│   ├── requirements.txt
│   ├── core/                # 纯逻辑，不依赖 FastAPI
│   │   ├── ir.py            # 统一中间表示 Canonical IR
│   │   ├── io/              # 图像 / 哈希 / 坐标工具
│   │   ├── ingest/          # 各格式接入适配器 + 多来源合并
│   │   ├── taxonomy/        # 类别体系规范化
│   │   ├── clean/           # 清洗规则引擎
│   │   ├── split/           # 数据集划分（分组防泄漏 / 分层）
│   │   ├── export/          # 导出（检测布局 + 分类布局）
│   │   ├── analytics/       # 统计与质量报告
│   │   ├── registry/        # 模型库（数据集血缘由 dataset_card.json 承载）
│   │   ├── train/           # 训练调度与指标解析
│   │   ├── eval/            # 评估调度、结果归一化、报告、多模型对比
│   │   └── deploy/          # 导出（格式探测 / 产物管理与校验）
│   ├── app/                 # FastAPI 层
│   ├── cli/                 # 命令行入口
│   ├── tests/               # 测试与夹具
│   └── storage/             # 生成的数据集 / 训练产物（运行时）
└── frontend/                # React + Ant Design
```

---

## 支持的输入格式

| 格式 | 适配器 | 标注形态 | 状态 |
|---|---|---|---|
| YOLO txt + `data.yaml`（Roboflow / ultralytics） | `yolo` | 目标检测（bbox） | 已实现 |
| COCO JSON（多文件或单文件） | `coco` | 目标检测 | 已实现 |
| Pascal VOC XML（含 `ImageSets/Main` 划分） | `voc` | 目标检测 | 已实现 |
| LabelMe JSON（矩形 / 多边形 / 圆形） | `labelme` | 检测（多边形保留分割信息） | 已实现 |
| OpenLABEL / VCD JSON（DMD 驾驶员监控数据集） | `openlabel` | 图像分类（逐帧动作，无框） | 已实现 |

> 不同来源的标注形态可能不同（检测 vs 分类）。IR 用 `Annotation.kind` 区分，
> 导出时会分别落成 YOLO 的检测布局（`images/` + `labels/`）或分类布局（`{split}/{class}/`）。
> 混合形态的数据集需先用类别规范化的 `kind_filter` 过滤（见 `core/taxonomy`）。

格式自动探测；也可用 `fmt` 参数强制指定。

### 数据流水线

```
多个来源（任意格式）
      │  ingest：各适配器解析
      ▼
   统一 IR（绝对像素 xyxy + 类名字符串 + kind）
      │  merge：合并、去重、类别并集
      │  taxonomy：类别映射/合并、按标注形态过滤、类别重排
      │  clean：图像损坏 / 重复 / 标签异常 / 跨子集泄漏
      │  split：分组防泄漏 + 分层 + 按体积分配 + 固定种子
      ▼
   export：出口做唯一一次归一化
      ├── 检测 → images/ + labels/ + data.yaml
      └── 分类 → {train,val,test}/{类别}/ + data.yaml
      ▼
   dataset_card.json（血缘：来源 / 划分参数 / 种子 / 统计）

分析（analytics）可作用于任意阶段的 IR：类别分布、目标尺寸分档、
子集构成偏移、单文件 HTML 报告。
```

合并多个来源时会把各自解析结果并入同一 IR，随后**统一划分与导出**，
这就是「不同格式数据合成同一个训练集/验证集/测试集」。

---

## 训练流水线（M2）

```
已生成的数据集（data.yaml + images/ + labels/）
      │  TrainSpec：一次训练的全部参数（写入 run_dir/train_spec.json）
      ▼
  subprocess 隔离执行 core.train.runner
      │   ├── backend.build_command()   组装命令行（不 import 重型依赖）
      │   └── backend.train()           子进程内调用 ultralytics YOLO.train()
      ▼
  run_dir/  ── results.csv   逐轮指标（唯一指标来源，不 hook ultralytics）
            ├── train.log    训练日志（stdout+stderr，落盘可回溯）
            ├── train_batch*.jpg / val_batch*.jpg / 曲线图 / 混淆矩阵
            ├── weights/best.pt, last.pt
            └── job.json     任务状态（服务重启后据此恢复任务列表）
      │
      ▼
  监控线程：读 results.csv（指标）+ psutil/pynvml（资源）+ 日志增量
      │
      ▼
  WebSocket 增量推送 → 训练详情页（实时曲线 / 资源 / 日志）
```

要点：

- **指标不靠 hook**：只读训练目录的 `results.csv`，升级 ultralytics 不会把可视化搞坏。
  解析器按列名取值，检测（`metrics/mAP50(B)`）与分类（`metrics/accuracy_top1`）都能用。
- **进程隔离**：训练崩溃、OOM 都不影响 API 进程；停止会终止整个进程树。
- **断点续训**：`resume` 复用同一训练目录与 `last.pt`，指标按 epoch 去重后继续累积。
- **无 GPU 降级**：CPU 数据始终可用；GPU 不可用时返回 `available=false` 并说明原因，
  界面只显示 CPU 面板。可选安装 `pynvml` 才有 GPU 利用率。
- **API 进程不加载 torch/ultralytics**：`build_command()` / 训练参数组装都是纯函数，
  重型依赖只在训练子进程里 import（测试中用一个子进程断言保证这一点）。
- **权重**：`.yaml` 结构文件表示从零训练（**离线可用**，不需要下载）；
  `.pt` 优先在 `YOLO_STUDIO_WEIGHTS` 目录按文件名查找，找不到才交给 ultralytics 下载。

训练任务的状态机：

```
pending → running → finished
              ├── → failed          （非零退出码）
              ├── → stopping → stopped（手动停止）
              └── 进程失联（服务重启）→ interrupted →（可）resume → running
```

训练接口（`app/api/routes/train.py`）：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/train/backends` | 训练后端与监控能力、默认配置 |
| POST | `/api/train/jobs` | 新建并启动训练 |
| GET | `/api/train/jobs` | 任务列表 |
| GET | `/api/train/jobs/{id}` | 详情（状态 + 指标 + 资源 + 日志尾部） |
| POST | `/api/train/jobs/{id}/stop` | 停止 |
| POST | `/api/train/jobs/{id}/resume` | 从 last.pt 断点续训 |
| GET | `/api/train/jobs/{id}/metrics` | 指标（支持 `since_epoch` 增量拉取） |
| GET | `/api/train/jobs/{id}/logs` | 日志（按 `offset`/`limit` 翻页） |
| GET | `/api/train/jobs/{id}/resources` | 资源采样 |
| GET | `/api/train/jobs/{id}/artifacts` | 过程图像清单 |
| GET | `/api/train/jobs/{id}/image` | 过程图像文件（受限访问） |
| WS | `/api/train/jobs/{id}/ws` | 实时增量事件（snapshot / metrics / log / resources / status / finished） |

> 训练前端页面（M5-05 详情页、M5-07 列表/新建页）见下文「前端界面（M5）」。

---

## 评估与模型库（M3）

```
训练完成（finished）
      │  自动注册：storage/models/<job_id>/model_card.json
      │           权重路径 + 训练超参 + 数据集血缘（反查 dataset_card.json）
      ▼
自动评估：val / test（有 test 用 test，否则 val）
      │  独立子进程 core.eval.runner → ultralytics model.val()
      ▼
storage/evals/<eval_id>/
      ├── eval_result.json   总体指标 + 逐类指标 + 混淆矩阵 + 耗时
      ├── eval.log
      └── confusion_matrix.png / BoxPR_curve.png / val_batch0_pred.jpg ...
      │  评估结束 → 结果挂到对应模型卡片上
      ▼
模型库：列表 / 详情 / 对比 / 单文件 HTML 评估报告
```

要点：

- **评估也是独立子进程**，与训练同样隔离；结果文件是唯一事实来源。
- **指标不凭空合成**：逐类指标、混淆矩阵、耗时都直接来自 `model.val()` 的公开返回值；
  解析不出的指标**不写入**，绝不用 0 冒充（0 会被误读为「效果很差」）。
- **对比必须先同划分**：val 与 test 的数字放在一起比较是错的，
  因此对比接口会检查划分是否一致并在结果里明确提示。
- **模型卡不复制数据**：只记录权重路径与评估目录引用，
  避免「卡片里的指标」与「评估目录里的结果」两处不一致。

评估与模型库接口：

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/eval/jobs` | 新建评估（给 `job_id` 或 weights+data_yaml） |
| GET | `/api/eval/jobs` / `/{id}` | 列表 / 详情（状态 + 结果 + 日志） |
| POST | `/api/eval/jobs/{id}/stop` | 停止 |
| GET | `/api/eval/jobs/{id}/result` | 结构化结果（逐类指标 + 混淆矩阵） |
| GET | `/api/eval/jobs/{id}/artifacts` / `/image` | 过程图像（PR/F1 曲线、混淆矩阵、样例预测） |
| POST | `/api/eval/jobs/{id}/report` | 生成单文件 HTML 评估报告 |
| WS | `/api/eval/jobs/{id}/ws` | 实时日志与状态 |
| POST | `/api/models/register` | 把已完成的训练任务注册为模型 |
| GET | `/api/models` / `/{model_id}` | 模型列表 / 详情（含血缘与评估索引） |
| GET | `/api/models/{model_id}/result` | 该模型某划分下最新评估结果 |
| POST | `/api/models/{model_id}/eval` | 对该模型发起评估 |
| POST | `/api/models/compare` | 多模型对比（总体 + 逐类 AP50-95） |

---

## 部署导出（M4）

```
模型库里的 best.pt（或任意权重）
      │  DeploySpec：格式、imgsz、精度、opset...
      ▼
  能力探测（find_spec，不加载重依赖）
      │  不可用的格式在启动前就拒绝，并说明缺哪个包 / 是否需要 GPU
      ▼
  subprocess 隔离执行 core.deploy.runner
      │  逐个格式导出，互不影响（部分成功也算完成）
      ▼
storage/deploys/<deploy_id>/
      ├── deploy_result.json   每个产物的路径 / 大小 / sha256
      ├── deploy.log
      └── best.onnx / best.torchscript ...（从权重旁复制过来）
      │  导出结束 → 产物挂到模型卡片（model_card.deploys）
      ▼
verify：重新计算大小与 sha256，与导出时的基线比对
```

要点：

- **格式能力透明**：`/api/deploy/formats` 逐格式说明「可用 / 缺什么」。
  不装了 onnx 却说支持 ONNX，是最容易让用户白等的事。
- **单格式失败不拖累其它格式**：只要有一个成功，任务就算完成，失败原因记在对应产物上。
- **产物先复制再校验**：ultralytics 默认把导出文件写在权重旁边，而训练目录可能被清理，
  因此默认复制到导出目录（`copy_artifacts=false` 可关闭）。
- **校验只谈文件完整性**：存在性 / 大小 / sha256。精度要重新推理，那是 M3 评估的职责。
- **RKNN（M4-02）**：需要 `rknn-toolkit2`，通常只在 Linux + 指定 Python 版本下可用；
  本机缺失时会在启动前明确拒绝，并提示按 rknn_model_zoo 的说明在独立环境导出。

导出接口：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/deploy/formats` | 格式清单与本机可用性 |
| POST | `/api/deploy/jobs` | 新建导出（给 `model_id` / `job_id` / `weights`） |
| GET | `/api/deploy/jobs` / `/{id}` | 列表 / 详情（含产物清单） |
| POST | `/api/deploy/jobs/{id}/stop` | 停止 |
| GET | `/api/deploy/jobs/{id}/result` | 结构化结果（产物 + 大小 + sha256） |
| GET | `/api/deploy/jobs/{id}/verify` | 产物完整性校验 |
| GET | `/api/deploy/jobs/{id}/download?fmt=onnx` | 下载指定格式产物 |
| WS | `/api/deploy/jobs/{id}/ws` | 实时日志与状态 |

> 部署导出的前端界面由 M5-08 的模型详情页承载（格式可用性、发起导出、产物下载与校验）。

---

## 前端界面（M5）

```
总览 / 数据导入 / 数据浏览 / 质量报告 / 数据集版本
训练任务列表 → 新建训练 → 训练详情（实时曲线 + 资源 + 日志）
模型库 → 模型详情（概览 / 评估 / 导出部署）→ 多模型对比
设置（只读：存储路径、运行环境、可选依赖与导出能力、访问白名单）
```

要点：

- **实时靠 WebSocket，不轮询**：训练详情页订阅 `/api/train/jobs/{id}/ws`。
  首个 `snapshot` 事件给全量状态，之后是 `metrics` / `log` / `resources` / `status` / `finished` 增量。
- **无 GPU 降级**：资源面板在 `gpu.available=false` 时只显示 CPU，并把原因（缺 pynvml / 无设备）如实展示。
- **缺失指标留空**：逐类指标、对比表、模型卡片里没有的指标一律显示占位符，绝不显示成 0。
- **对比先同划分**：模型对比页展示各结果使用的划分，划分不一致时后端会给出提示。
- **数据浏览只读**：`/api/datasets/browse` 在内存 IR 上筛选（类别 / 划分 / 来源 / 形态 / 目标数 / 长边），
  缩略图按原图坐标叠加标注框，不修改任何原始文件。
- **按需图表 + 按页拆包**：所有页面用 `React.lazy` 做路由级懒加载，首屏只加载当前页所需代码；
  ECharts 只注册实际用到的折线 / 柱状 / 热力图（`components/echarts.tsx`），echarts chunk 由 ~1.0MB 降到 ~0.58MB。
- **配置只读呈现**：设置页数据来自 `GET /api/system/config`，展示每个路径由哪个 `YOLO_STUDIO_*`
  环境变量控制；配置在后端启动时读取，运行期不可改，改完需重启（不做“改了不生效”的假编辑）。

---

## 核心设计决策

### 1. 统一中间表示（Canonical IR）先于一切

所有格式先归一到**绝对像素坐标 `xyxy` + 类别字符串名**，只在**导出阶段做一次** YOLO 归一化。

> 中间过程绝不来回转换坐标，否则精度损失与类别错位极难排查。

### 2. 类别以「名字」为准，而不是 id

不同数据集的 `class 0` 含义完全不同。映射必须按**类名字符串 + 别名表**做，最后才统一分配下标。

### 3. 清洗规则可 dry-run

每条规则产出「问题清单 + 处置建议」，人工确认后才落盘。任何删除都可回溯。

### 4. 防数据泄漏是内建能力

- train/val/test 之间的**重复图检测**（近似重复用 pHash）
- **分组划分**：同一视频/序列的帧必须整体落在同一个子集

### 5. 数据集版本不可变

每次合成为一个版本 `v1, v2...`，记录输入源、清洗规则、映射表、随机种子。模型可反查"我是在哪版数据上训的"。

### 6. 环境与部署

- 训练依赖（ultralytics / torch）声明在 `backend/requirements.txt`，随 venv 安装；
- 所有路径基于项目内相对位置，可用环境变量覆盖；
- 训练解释器默认当前解释器，用 `YOLO_STUDIO_PYTHON` 可指定其他环境。

### 7. 出口只做一次归一化

绝对像素坐标 → YOLO 归一化坐标**只在导出时发生一次**。
中间过程（接入/合并/清洗/划分）全部保持绝对像素，避免来回转换造成的精度损失与类别错位。

### 8. 合并冲突绝不静默丢弃

多来源复制同名文件是常态。合并时 uid 冲突会分两种情况处理：
同一文件重复导入 → 跳过并记录；不同文件撞车 → 重新分配 uid 并同步改写标注引用。

### 9. 训练只从训练目录读结果，不 hook 框架内部

指标读 `results.csv`、图像读训练目录里的过程图、日志读子进程 stdout。
训练框架被当作「黑盒 + 约定产物目录」，因此升级 ultralytics 不会连累可视化。
代价是少数内部状态看不到——这是有意接受的取舍。

### 10. 训练参数走文件，不走命令行

`TrainSpec` 序列化到 `run_dir/train_spec.json`，命令行只传这个文件路径。
好处有三：参数再多也不撞 Windows 命令行长度限制；事后追溯「这次训练用了什么」只看一个文件；
指标解析与调度都不需要理解具体框架的参数。

### 11. 缺失的指标不补 0

评估结果里，解析不出来或为 NaN 的指标**直接不写入**。
把「没有这个指标」写成 0 会被读成「效果为 0」，比缺少字段危险得多。
逐类对比表同理：某个模型没有该类时留空。

### 12. 对比前先确认划分一致

val 与 test 上的指标不具备可比性。对比接口会收集各结果使用的划分，
不一致时在 `notes` 里明确提示，而不是给出一张看起来正常、实则误导的表格。

### 13. 能力探测要如实，不要乐观

导出格式、GPU 监控、可选依赖——凡是有可能缺失的能力，都先探测再暴露，
并在不可用时给出**具体原因**（缺哪个包、是否需要 GPU）。
乐观地"假设可用"只会让用户在失败后自己猜原因。

### 14. 产物要可校验

导出的每个产物都记录大小与 sha256，提供 `verify` 重新比对。
模型文件被截断、被替换、被误删是部署阶段最常见的故障，
有基线才能快速判断"是文件坏了还是模型本身有问题"。

---

## 快速开始

见 [docs/getting-started.md](docs/getting-started.md)。（M0~M5 已完成，见 `PROGRESS.md`）

---

## 进度

所有模块与任务状态见 [`PROGRESS.md`](PROGRESS.md)。
新会话/新协作者请先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)（协作约定、环境事实、续作提示）。
