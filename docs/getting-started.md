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
    --weights yolov8n.yaml --epochs 1 --imgsz 32 --batch 4 --device cpu --dry-run

# 真正训练（CPU）
.\.venv\Scripts\python.exe -m cli.main train `
    --data "storage\datasets\dms_grouped_demo\data.yaml" `
    --weights yolov8n.yaml --epochs 5 --imgsz 320 --device cpu --name my_run

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
.\.venv\Scripts\python.exe tests\test_ingest.py     # 63  接入层（含 /api/datasets/browse）
.\.venv\Scripts\python.exe tests\test_pipeline.py   # 89  划分 + 导出 + 多源合并
.\.venv\Scripts\python.exe tests\test_clean.py      # 58  清洗引擎
.\.venv\Scripts\python.exe tests\test_taxonomy.py   # 69  类别规范化
.\.venv\Scripts\python.exe tests\test_adapters.py   # 55  COCO / VOC / LabelMe
.\.venv\Scripts\python.exe tests\test_analytics.py  # 84  统计分析
.\.venv\Scripts\python.exe tests\test_train.py      # 142 训练模块（调度/指标/日志/资源/产物/API/WS）
.\.venv\Scripts\python.exe tests\test_eval.py       # 174 评估与模型库（归一化/报告/注册/外部导入/对比/API）
.\.venv\Scripts\python.exe tests\test_deploy.py     # 103 部署导出（格式探测/哈希校验/生命周期/API/联动）
.\.venv\Scripts\python.exe tests\test_infer.py      # 97  M6 实时验证（格式/归一化/会话/协议/API/背压）
.\.venv\Scripts\python.exe tests\test_prelabel.py   # 97  M7 预标注（收集/IR/统计/失败跳过/分类/血缘/导出卡片）
```

> 全量共 **1031 断言**（11 个文件）。`test_train` 存在一个**环境相关的偶发失败**
> （约 1/8~1/12，全量连跑时更高，`wait_terminal` 25s 超时；见 `PROGRESS.md` R-47），
> 与功能无关，偶发时重跑即可。
>
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

## 五、部署到另一台机器 / 局域网共享（含 GPU 机器）

目标场景：**后端 + 前端跑在一台（有 GPU 的）机器上，你从另一台机器用浏览器访问**。
这本来就是本项目支持的标准用法——后端在生产模式下会直接托管前端。

### 推荐：单端口生产模式（最省事）

在**服务器那台机器**上：

```powershell
# 1) 前端先构建一次（产物 frontend/dist 会被后端托管）
cd frontend; npm run build

# 2) 启动后端（监听所有网卡，不带热重载）
powershell -ExecutionPolicy Bypass -File scripts\start-server.ps1
#    等价于： cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

然后在客户端浏览器访问 `http://<服务器IP>:8010/`。

**为什么推荐**：前端与 API 同源 —— 没有 CORS 问题，WebSocket 自动走同源，
`/train/:id` 这类深链刷新也正常，图片/文件接口同源。`scripts/start-server.ps1` 默认
`-BindHost 0.0.0.0 -Port 8010`，可传参覆盖。

### 开发模式两端口（Vite 5173 + 后端 8010）

```powershell
powershell -File scripts\start-backend.ps1 -BindHost 0.0.0.0
powershell -File scripts\start-frontend.ps1 -BindHost 0.0.0.0
```

这种模式用 `http://<服务器IP或域名>:5173` 访问前端。默认前端通过 **Vite 代理**调用 `/api`
（浏览器看到的仍是 5173，同源）——**同源就不会触发 CORS，因此通常不需要配 CORS**。
只有当你不走代理、让前端**直连后端**时（设 `VITE_API_BASE=http://<IP>:8010`，浏览器直接请求 8010），
才需要在后端放行该来源：

```powershell
# 仅当 VITE_API_BASE 指向后端、浏览器直连 8010 时才需要
$env:YOLO_STUDIO_CORS_ORIGINS = "http://192.168.1.20:5173"   # 分号或逗号分隔；重启后端生效
```

> **Vite 的主机校验**（`server.allowedHosts`，5.4.12+ 的安全补丁）：**IP 地址（含公网 IP）默认就放行**，
> 不需要配置；只有用**主机名/域名**访问（如 `http://my-pc:5173`、隧道域名）才会返回
> `403 Blocked request. This host ("my-pc") is not allowed.`，此时用 `VITE_ALLOWED_HOSTS` 放行
> （会读取 `frontend/.env.local`）：
>
> ```powershell
> # 填主机名/域名（不是 IP！）；逗号分隔；重启前端 dev server 生效
> $env:VITE_ALLOWED_HOSTS = "my-pc,my-pc.lan,dev.example.com"
> # 特殊值 true / all / * = 放行任意主机（关闭该防护，仅限可信内网）
> ```


### 必做的几件事

| 项 | 说明 |
|---|---|
| 监听地址 | 后端 `--host 0.0.0.0`（`start-server.ps1` 已默认） |
| 防火墙 | Windows Defender 防火墙放行 8010/TCP |
| **读取白名单** | `YOLO_STUDIO_ALLOWED_ROOTS="D:\datasets;D:\models"`。默认**空 = 不限制**，任何人都能读服务器上任意文件 |
| **认证** | 本项目**没有登录/鉴权**。只在内网/VPN 用，或在前面套反代（Nginx/Caddy）做 Basic Auth |
| CORS | 默认走 Vite 代理 = 同源，**不需要**；仅当 `VITE_API_BASE` 让前端直连后端时才需要（见上）。单端口同源也不需要 |
| 请求超时 | 数据管道的 `scan`/`clean`/`split`/`export` 是**同步**接口，可能跑好几分钟。前端默认已放宽到 **5 分钟**（`VITE_API_TIMEOUT`）；用反代时 **nginx 的 `proxy_read_timeout` 默认只有 60s，必须一起调**，否则最先断的是它（见第六节） |

### 摄像头（实时验证）需要 HTTPS

浏览器要求 `getUserMedia` 运行在**安全上下文**（HTTPS 或 `localhost`）。用
`http://<服务器IP>:8010` 访问时，**实时验证页的摄像头会被浏览器禁用**；图片上传/服务器路径推理
以及预标注、训练、评估、导出、模型库等全部不受影响。想远端用摄像头就配 HTTPS
（自签证书 / 反代终止 TLS / 隧道）。
注意摄像头用的是**你本地浏览器**的摄像头，服务器机器不需要摄像头。

### 环境与数据

- 服务器上装 **CUDA 版 torch**：`scripts\install-gpu.ps1` 一条命令搞定（见「训练环境」），
  `YOLO_STUDIO_DEVICE=auto` 会自动用 GPU；`nvidia-ml-py`（GPU 利用率面板）由脚本一起装。
- 血缘/路径都是写在 JSON 里的**绝对路径**。把别处的 `storage/` 直接拷过来会因盘符/路径不同而失效：
  建议在服务器上重新接入数据，或用 `YOLO_STUDIO_*` 把各目录指到实际位置后重新注册。
- 关掉浏览器不影响训练/评估（服务端子进程）；但**后端进程要常驻**，重启后能接管仍在跑的进程。

---

## 六、大数据集（十万张量级）

数据管道的五个接口 —— `scan` / `clean` / `split` / `export` / `browse` —— 是**同步**的：
在请求里跑完才返回，**没有后台任务、没有进度**（训练/评估/部署/预标注则相反，是子进程 + WebSocket 进度，
不受数据集大小影响）。另外 `/api/datasets/browse` 每次请求都会**重新读一遍全量数据**，没有缓存。

实测每张图的边际耗时（本机，640×480，扣掉进程启动开销；真实图更大，磁盘 I/O 部分会更高）：

| 步骤 | 每张 | 10 万张 | 20 万张 |
|---|---|---|---|
| `scan`（接入 IR） | 0.94 ms | ≈ 1.6 min | ≈ 3.1 min |
| `clean`（解码校验 + 完全重复） | 2.52 ms | ≈ 4.2 min | ≈ 8.4 min |
| `clean --no-verify` | 1.50 ms | ≈ 2.5 min | ≈ 5 min |
| `export --file-mode hardlink` | 2.21 ms | ≈ 3.7 min | ≈ 7.4 min |

> 上表的 `clean` 关掉了近似重复检测（`--no-near`）；**默认是开着的，会更慢**。

### 超时：一处配置

前端 axios 与 Vite 开发代理都读同一个环境变量，默认 **5 分钟**：

```powershell
$env:VITE_API_TIMEOUT = "900000"   # 毫秒；前端 dev server 与构建都读它
```

- 改了要**重启 vite**（dev）或**重新 `npm run build`**（生产，值在构建时写进产物）；
- 用了反代的话，nginx 侧还要 `proxy_read_timeout 900s;`（默认 60s，最先断的就是它）；
- 生产单端口模式（后端托管前端）只有 axios 这一处。

### 建议：十万张以上走 CLI

同步接口在大数据集上的体验就是「长时间白屏」，所以处理这种包用命令行更稳（有输出、没有 HTTP 超时）：

```powershell
cd backend
# 体检（默认 dry-run，不改数据），--no-near 可先快速过一遍
.\.venv\Scripts\python.exe -m cli.main clean "<数据集根>" --no-near --show 20
# 合成 + 划分 + 导出；hardlink 不额外占盘（要求与源数据同一块盘）
.\.venv\Scripts\python.exe -m cli.main export "<数据集根>" --out "E:\datasets\_studio\ds_v1" --file-mode hardlink
```

两个配套提醒：

- **训练不必先有项目版本**：新建训练页的「数据集 data.yaml」是 AutoComplete，可以直接粘**任意绝对路径**
  （`TrainNew.tsx`），训练本身是后台子进程 + WebSocket 进度。
- **GPU 训练记得调 `workers`**：默认 `0`（Windows 保守值）会让数据加载喂不满 GPU，N 卡上建议 4~8；
  `cache` 默认不缓存，十万张量级不要开（除非内存足够）。
- 本机实测过的最大规模是 **9884 图 / 23389 框**（DMS 检测数据集），更大规模请按上表估算时间。

---

## 训练环境

训练依赖（`ultralytics` + `torch` + `torchvision`）已声明在 `requirements.txt` 中。

### 让项目检测到 GPU（装 CUDA 版 torch）

**Windows 上 `uv pip install -r requirements.txt` 装的是 CPU 版 torch**（PyPI 默认），
所以有 N 卡的机器装完 `torch.cuda.is_available()` 仍是 `False`。用脚本按驱动支持的版本重装：

```powershell
# 先看要做什么（不做任何改动）
powershell -ExecutionPolicy Bypass -File scripts\install-gpu.ps1 -DryRun
# 执行
powershell -ExecutionPolicy Bypass -File scripts\install-gpu.ps1
```

脚本的步骤：读 `nvidia-smi` 的驱动 / CUDA 版本与显卡的 compute capability → **先卸掉 CPU 版 torch**
（不卸的话 uv 会认为版本号没变，直接跳过）→ 按对应 index 装 CUDA 版 → 装 `nvidia-ml-py`
（训练详情页的 GPU 利用率面板）→ 跑一次真实矩阵乘法确认 kernel 可用。

- **RTX 50 系是 Blackwell（compute capability 12.0），必须 cu128 及以上**。cu126/cu124 的 wheel
  不含 sm_120 kernel：`cuda_available` 会是 `True`，但一开训练就报
  `no kernel image is available for execution on the device`。脚本检测到这种组合会直接拒绝，
  并提示升级驱动。
- 装完必须**重启后端**（`--reload` 只重载源码，换 torch 这种二进制不够）。
- 反向注意：日常别用 `uv pip install -r requirements.txt --reinstall`，它会把 torch 换回 CPU 版。

不用脚本时的手动等价命令：

```powershell
cd backend
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip uninstall torch torchvision            # 必须先卸：否则 uv 认为版本没变而跳过
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install nvidia-ml-py                   # GPU 利用率面板（可选）
# 也可用 cu121 / cu118 / cpu 等其它 index
```

Linux 上 PyPI 默认 torch 就带 CUDA，通常不需要这一步；要指定版本时把 `--index-url` 换成
`cu121` / `cu118` 等即可。

### 训练解释器与设备

- 解释器默认当前解释器，需要指定其他环境时用 `YOLO_STUDIO_PYTHON`；
- 设备：`YOLO_STUDIO_DEVICE=auto`（默认，有 CUDA 用 GPU 否则 CPU）/ `cpu` / `cuda:0`。

开发用的本机是纯 CPU 环境（`torch.cuda.is_available()` = **False**），训练以 CPU 运行，
前端可视化自动降级为不显示 GPU 面板；换到有 N 卡的机器后按上一节装 CUDA 版 torch 即可。

### 权重与离线

- `.yaml` 结构文件（如 `yolov8n.yaml` / `yolov8-cls.yaml`）表示**从零训练**，
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
| `YOLO_STUDIO_REPORTS` | `backend/storage/reports` | 质量报告输出目录 |
| `YOLO_STUDIO_DB` | `backend/storage/studio.db` | **保留字段，当前未使用**（本项目用文件系统而非数据库，见 README 决策 R-10） |
| `YOLO_STUDIO_PORT` | `8010` | API 监听端口（避免与 8000 的 LLM 网关冲突） |
| `YOLO_STUDIO_CORS_ORIGINS` | `http://localhost:5173;http://127.0.0.1:5173` | CORS 允许来源（分号或逗号分隔）。从别的机器/域名访问时把它加进来；同源单端口部署无需配置 |
| `YOLO_STUDIO_PYTHON` | 当前解释器 | 训练使用的解释器 |
| `YOLO_STUDIO_DEVICE` | `auto` | 训练设备：`auto` / `cpu` / `cuda:0` |
| `YOLO_STUDIO_ALLOWED_ROOTS` | 空（不限制） | 图片/文件读取白名单，分号分隔。**对外提供服务时必须设置** |
