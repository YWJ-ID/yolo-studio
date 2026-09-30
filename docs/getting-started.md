# 快速开始

## 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.9+ | 项目自带 `backend/.venv`（3.9.25） |
| Node.js | 18+ | 本机 v20.19.0 |
| 训练环境 | ultralytics 8.x | 见下方「训练环境」 |

## 目录说明

```
yolo-studio/
├── backend/     后端（FastAPI + core 纯逻辑层）
└── frontend/    前端（React + Ant Design + Vite）
```

前后端完全分离：开发时前端跑 5173，通过 Vite 代理访问后端 8010。

---

## 一、启动后端

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
```

- 健康检查：<http://127.0.0.1:8010/api/system/health>
- 接口文档：<http://127.0.0.1:8010/docs>

> 默认端口 **8010**（不用 8000：本机 8000 已被 OpenAI 兼容网关占用）。
> 改端口：`$env:YOLO_STUDIO_PORT="8020"`，同时改 `frontend/vite.config.ts` 的 proxy target。

首次部署（已有 `.venv` 可跳过）：

```powershell
uv venv --python 3.9 .venv
$env:VIRTUAL_ENV="<项目绝对路径>\backend\.venv"
uv pip install -r requirements.txt
```

## 二、启动前端

```powershell
cd frontend
npm install     # 首次
npm run dev     # http://localhost:5173
```

生产构建：

```powershell
npm run build   # 产物在 frontend/dist，后端会自动托管
```

---

## 三、命令行（不依赖 Web，验证 core）

```powershell
cd backend
.\.venv\Scripts\python.exe -m cli.main adapters          # 列出支持的格式
.\.venv\Scripts\python.exe -m cli.main detect <目录>      # 探测格式
.\.venv\Scripts\python.exe -m cli.main scan   <目录>      # 扫描统计
.\.venv\Scripts\python.exe -m cli.main scan   <目录> --json
```

示例：

```powershell
# 单来源
.\.venv\Scripts\python.exe -m cli.main scan "D:\dataset\archive"

# 多来源合并（不同格式可混合，各自自动探测）
.\.venv\Scripts\python.exe -m cli.main scan "D:\dataset\archive" "D:\dmd_session"

# 合并 + 划分（只看分布，不落盘）
.\.venv\Scripts\python.exe -m cli.main split "D:\dataset\archive" "D:\dmd_session" --split-ratio 0.8,0.1,0.1

# 合并 + 清洗 + 划分 + 导出为可训练数据集
.\.venv\Scripts\python.exe -m cli.main export "D:\dataset\archive" "D:\dmd_session" `
    --out "storage\datasets\merged" --clean --file-mode hardlink --overwrite
```

### 类别规范化

```powershell
# 查看类别清单、同义类名建议、类名合法性问题（不做改动）
.\.venv\Scripts\python.exe -m cli.main taxonomy --suggest "D:\dataset\archive"

# 合并同义类名
.\.venv\Scripts\python.exe -m cli.main taxonomy "D:\dataset\archive" `
    --map "closed_eye=Closed Eye" --map "closed-eye=Closed Eye"

# 只保留检测框（混合形态数据集无法用单一布局导出，必须先选定一种形态）
.\.venv\Scripts\python.exe -m cli.main taxonomy "D:\dataset\archive" --kind bbox

# 显式指定类别顺序（用旧权重继续训练时必须与旧模型一致）
.\.venv\Scripts\python.exe -m cli.main taxonomy "D:\dataset\archive" --order "Open Eye,Closed Eye,Cigarette,Phone,Seatbelt"
```

在 `export` 上也可直接用同样的参数，导出流程顺序为
**类别规范化 → 清洗 → 划分 → 导出**。

### 统计分析与质量报告

```powershell
# 终端查看统计与结论
.\.venv\Scripts\python.exe -m cli.main report "D:\dataset\archive"

# 导出单文件 HTML 报告（内嵌图表与缩略图，可离线分享）
.\.venv\Scripts\python.exe -m cli.main report "D:\dataset\archive" `
    --out "storage\reports\archive.html" --sample 36

# 只要数据不要报告体积
.\.venv\Scripts\python.exe -m cli.main report "D:\dataset\archive" --out report.html --no-embed

# 全部统计结果以 JSON 输出
.\.venv\Scripts\python.exe -m cli.main report "D:\dataset\archive" --json
```

报告包含：类别分布与不均衡量化、目标尺寸分档（对应 P3/P4/P5 检测层）、
标注框面积占比、每图目标数、类别 × 子集矩阵、图像尺寸分布、
**子集类别构成偏移检测**，以及由这些事实推导的处置建议。

### 数据清洗

```powershell
# 只检查，不修改任何数据（dry-run）
.\.venv\Scripts\python.exe -m cli.main clean "D:\dataset\archive"

# 列出全部规则
.\.venv\Scripts\python.exe -m cli.main clean --list-rules

# 查看问题明细
.\.venv\Scripts\python.exe -m cli.main clean "D:\dataset\archive" --show 10

# 快速预览（只查前 500 张，跳过慢检查）
.\.venv\Scripts\python.exe -m cli.main clean "D:\dataset\archive" --limit 500 --no-verify --no-near
```

清洗作用在解析出的中间表示上，**不会修改原始数据文件**；
`export --clean` 会在导出前自动执行处置（裁剪越界框、删除重复图与跨子集泄漏样本）。

### 分组防泄漏（视频抽帧数据必看）

视频相邻帧几乎完全相同，一旦跨 train/val，验证指标会严重虚高。
用 `--group-by` 指定如何识别「同一段视频」：

```powershell
--group-by parent                    # 按子目录分组
--group-by stem                      # 按文件名末尾数字前的前缀
--group-by "regex:^(.*?)_mp4-\d+"    # 用正则提取（第一个捕获组为组名）
```

常用导出参数：

| 参数 | 说明 |
|---|---|
| `--task` | `auto`（默认）/ `detection` / `classification` |
| `--file-mode` | `copy`（默认，最通用）/ `hardlink`（需同盘，不占额外空间）/ `symlink` |
| `--split-ratio` | 默认 `0.8,0.1,0.1` |
| `--seed` | 划分随机种子，默认 42，可复现 |
| `--no-groups` | ⚠ 关闭分组防泄漏，视频抽帧数据**不要**用 |
| `--class-order` | 逗号分隔，强制类别下标顺序（用旧权重继续训练时很重要） |
| `--overwrite` | 允许覆盖非空输出目录 |

### 各格式的专用参数

| 格式 | 参数 | 说明 |
|---|---|---|
| COCO / VOC | `--images-dir` | 图像所在目录，默认自动推断 |
| VOC | `--voc-one-based` | 按 1-based 闭区间处理坐标（原始 VOC devkit）；默认按 LabelImg 的 0-based |
| OpenLABEL | `--level` | 取哪个标注层级作为类别（默认 `driver_actions`） |
| OpenLABEL | `--frames-dir` | 帧图片目录，默认自动探测 |
| OpenLABEL | `--include-objects` | 并入 `object` 类型的标注 |
| 通用 | `--group-by` | `none` / `parent` / `stem` / `regex:<正则>` |

### 训练（M2）

```powershell
# 只组装命令、不启动（用于确认参数与路径）
.\.venv\Scripts\python.exe -m cli.main train `
    --data "storage\datasets\dms_grouped_demo\data.yaml" `
    --weights yolo11n.yaml --epochs 1 --imgsz 32 --batch 4 --device cpu --dry-run

# 真正训练（CPU）
.\.venv\Scripts\python.exe -m cli.main train `
    --data "storage\datasets\dms_grouped_demo\data.yaml" `
    --weights yolo11n.yaml --epochs 5 --imgsz 320 --device cpu --name my_run

# 只启动不等待（适合长训练）
.\.venv\Scripts\python.exe -m cli.main train --data <data.yaml> --no-wait
```

| 参数 | 说明 |
|---|---|
| `--data` | 数据集 `data.yaml` 路径 |
| `--weights` | `.yaml` = 从零训练（离线可用）；`.pt` = 预训练权重；留空按 task 取默认 |
| `--task` | `detect`（默认）/ `classify` |
| `--device` | `cpu` / `cuda:0`；留空读 `YOLO_STUDIO_DEVICE` |
| `--workers` | dataloader 进程数，CPU 训练建议 0 |
| `--runs-dir` | 训练产物根目录，默认 `backend/storage/runs` |
| `--weights-dir` | 预训练权重目录，按文件名引用 `.pt` |
| `--no-wait` | 只启动不等待 |
| `--dry-run` | 只输出将执行的命令 |

真实训练冒烟（不需要联网）：

```powershell
.\.venv\Scripts\python.exe tests\fixtures\make_tiny_det.py
.\.venv\Scripts\python.exe -m cli.main train `
    --data storage\demo_sources\tiny_det\data.yaml `
    --weights yolo11n.yaml --epochs 1 --imgsz 32 --batch 4 --device cpu --name smoke_yolo11n
```

产物在 `backend/storage/runs/<name>/`：`results.csv`、`train.log`、过程图像、`weights/`。

### 训练接口

启动后端后（见「一」），`/api/train/*` 提供任务列表、新建、停止、续训、
指标、日志、资源、过程图像与 WebSocket 实时推送。接口文档见 <http://127.0.0.1:8010/docs>。

### 评估与模型库（M3）

训练**成功后会自动**：注册模型（`storage/models/<job_id>/model_card.json`）+ 在 test（没有则 val）上评估。
关掉自动评估：`$env:YOLO_STUDIO_AUTO_EVAL="0"`；指定划分：`$env:YOLO_STUDIO_AUTO_EVAL_SPLIT="val"`。

用接口操作（以真实冒烟模型为例）：

```powershell
# 手工注册一个已完成的训练任务（幂等）
Invoke-RestMethod -Uri http://127.0.0.1:8010/api/models/register -Method Post `
  -ContentType 'application/json' -Body '{"job_id":"smoke_yolo11n"}'

# 在 test 划分上评估
Invoke-RestMethod -Uri http://127.0.0.1:8010/api/models/smoke_yolo11n/eval -Method Post `
  -ContentType 'application/json' -Body '{"split":"test"}'

# 看模型列表 / 详情 / 对比
Invoke-RestMethod http://127.0.0.1:8010/api/models
Invoke-RestMethod http://127.0.0.1:8010/api/models/smoke_yolo11n
Invoke-RestMethod -Uri http://127.0.0.1:8010/api/models/compare -Method Post `
  -ContentType 'application/json' -Body '{"model_ids":["smoke_yolo11n","smoke_e3"],"split":"test"}'

# 生成单文件 HTML 评估报告（含逐类指标、混淆矩阵、曲线与样例预测）
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/eval/jobs/<eval_id>/report" -Method Post `
  -ContentType 'application/json' -Body '{"title":"模型评估报告"}'
```

> 对比只对**同一划分**的评估结果有意义；接口会在 `notes` 中提示划分不一致。

也可以只用命令行评估（不依赖 Web）：

```powershell
.\.venv\Scripts\python.exe -m cli.main eval `
    --weights storage\runs\smoke_yolo11n\weights\best.pt `
    --data storage\demo_sources\tiny_det\data.yaml --split test --imgsz 32 --device cpu
```

### 部署导出（M4）

先看本机支持哪些格式（会说明缺哪个包 / 是否需要 GPU）：

```powershell
.\.venv\Scripts\python.exe -m cli.main export-model --list-formats
```

```powershell
# 从权重导出（本机可用的只有 torchscript，因为没装 onnx）
.\.venv\Scripts\python.exe -m cli.main export-model `
    --weights storage\runs\smoke_e3\weights\best.pt `
    --formats torchscript --imgsz 32 --device cpu --name my_export
```

用接口导出（可从模型库直接取 best.pt）：

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/deploy/formats

# 从模型库导出
Invoke-RestMethod -Uri http://127.0.0.1:8010/api/deploy/jobs -Method Post `
  -ContentType 'application/json' `
  -Body '{"model_id":"smoke_e3","formats":["torchscript"],"imgsz":32}'

# 校验产物完整性（重新比对大小与 sha256）
Invoke-RestMethod "http://127.0.0.1:8010/api/deploy/jobs/<deploy_id>/verify"

# 下载产物
Invoke-WebRequest "http://127.0.0.1:8010/api/deploy/jobs/<deploy_id>/download?fmt=torchscript" -OutFile best.torchscript
```

**启用 ONNX / OpenVINO 等格式**（可选，按需安装）：

```powershell
uv pip install onnx onnxslim        # ONNX 导出（onnxslim 用于图简化）
uv pip install onnxruntime          # 想用 onnxruntime 做数值校验时
uv pip install openvino             # OpenVINO
```

> RKNN 需要 `rknn-toolkit2`，一般只在 Linux + 指定 Python 版本下可用，
> 通常按 rknn_model_zoo 的说明在独立环境里导出。

---

## 四、测试

```powershell
cd backend
.\.venv\Scripts\python.exe tests\test_ingest.py     # 接入层（49 断言）
.\.venv\Scripts\python.exe tests\test_pipeline.py   # 划分 + 导出 + 合并（89 断言）
.\.venv\Scripts\python.exe tests\test_clean.py      # 清洗引擎（58 断言）
.\.venv\Scripts\python.exe tests\test_taxonomy.py   # 类别规范化（69 断言）
.\.venv\Scripts\python.exe tests\test_adapters.py   # COCO / VOC / LabelMe（55 断言）
.\.venv\Scripts\python.exe tests\test_analytics.py  # 统计分析与报告（84 断言）
.\.venv\Scripts\python.exe tests\test_train.py      # 训练模块（142 断言）
.\.venv\Scripts\python.exe tests\test_eval.py       # 评估与模型库（147 断言）
.\.venv\Scripts\python.exe tests\test_deploy.py     # 部署导出（103 断言）
```

> Windows 控制台是 GBK，个别测试输出含 `²` 会抛 `UnicodeEncodeError`。
> 先设 `$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'` 再跑即可（不是测试失败）。

覆盖接入适配器、类别规范化、清洗、划分与导出、统计分析与训练调度。

其中 OpenLABEL 用到的夹具由官方 `vcd` 库生成，**已提交**，测试本身不依赖 `vcd` 包。
如需重新生成（需要 DMD 仓库自带的 3.8 环境）：

```powershell
& "<DMD仓库>\py38\Scripts\python.exe" backend\tests\fixtures\make_dmd_fixture.py
```

训练模块测试用 `fixtures/fake_trainer.py` 模拟 ultralytics 的外部产物（results.csv /
日志 / 过程图），因此不需要真跑 YOLO 就能验证调度、指标、日志与产物接口。

---

## 训练环境

训练依赖（`ultralytics` + `torch` + `torchvision`）已声明在 `requirements.txt` 中。

### 指定 CUDA 版本

`ultralytics` 会带上 PyPI 默认版 torch（Windows 默认 CPU 版，Linux 默认带 CUDA）。
若要明确指定，**先**单独装 torch，再装 requirements：

```powershell
# CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 强制 CPU
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

uv pip install -r requirements.txt
```

### 训练解释器与设备

- 解释器默认当前解释器，需要指定其他环境时用 `YOLO_STUDIO_PYTHON`；
- 设备：`YOLO_STUDIO_DEVICE=auto`（默认，有 CUDA 用 GPU 否则 CPU）/ `cpu` / `cuda:0`。

当前本机 `torch.cuda.is_available()` = **False**，训练以 CPU 运行，
前端可视化会自动降级为不显示 GPU 面板。

### 权重与离线

- `.yaml` 结构文件（如 `yolo11n.yaml` / `yolo11-cls.yaml`）表示**从零训练**，
  文件在 ultralytics 包内，**不需要联网**；
- `.pt` 预训练权重：先在 `storage/weights`（可用 `YOLO_STUDIO_WEIGHTS` 改）按文件名查找，
  找不到才交给 ultralytics 下载。

**真实训练已在本机验证**：yolo11n.yaml 从零、CPU、imgsz=32、1 epoch，
退出码 0 并产出 `results.csv` 与 12 张过程图；停止后续训也验证通过。

---

## 配置项

均可用环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `YOLO_STUDIO_STORAGE` | `backend/storage` | 运行产物根目录 |
| `YOLO_STUDIO_DATASETS` | `backend/storage/datasets` | 生成的数据集版本 |
| `YOLO_STUDIO_RUNS` | `backend/storage/runs` | 训练产物 |
| `YOLO_STUDIO_EVALS` | `backend/storage/evals` | 评估产物（指标 / 混淆矩阵 / 曲线） |
| `YOLO_STUDIO_MODELS` | `backend/storage/models` | 模型库（每个模型一份 `model_card.json`） |
| `YOLO_STUDIO_DEPLOYS` | `backend/storage/deploys` | 部署导出产物 |
| `YOLO_STUDIO_AUTO_EVAL` | `1` | 训练完成后是否自动评估（`0` 关闭） |
| `YOLO_STUDIO_AUTO_EVAL_SPLIT` | `auto` | 自动评估的划分：`auto`/`val`/`test`/`train` |
| `YOLO_STUDIO_WEIGHTS` | `backend/storage/weights` | 预训练权重目录（按文件名引用 `.pt`） |
| `YOLO_STUDIO_UPLOADS` | `backend/storage/uploads` | 上传的原始数据 |
| `YOLO_STUDIO_DB` | `backend/storage/studio.db` | sqlite 元数据库 |
| `YOLO_STUDIO_PORT` | `8010` | API 监听端口（避免与 8000 的 LLM 网关冲突） |
| `YOLO_STUDIO_PYTHON` | 当前解释器 | 训练使用的解释器 |
| `YOLO_STUDIO_DEVICE` | `auto` | 训练设备：`auto` / `cpu` / `cuda:0` |
| `YOLO_STUDIO_ALLOWED_ROOTS` | 空（不限制） | 图片读取白名单，分号分隔。**对外提供服务时必须设置** |
