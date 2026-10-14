# 让后端检测到 NVIDIA GPU：把 venv 里的 CPU 版 torch 换成 CUDA 版
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\install-gpu.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install-gpu.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\install-gpu.ps1 -Index cu128
#   powershell -ExecutionPolicy Bypass -File scripts\install-gpu.ps1 -Yes -SkipMonitor
#
# 为什么需要这个脚本：requirements.txt 里的 ultralytics 在 Windows 上带的是 PyPI
# 默认的 **CPU 版 torch**，装完 `torch.cuda.is_available()` 仍是 False，训练照旧跑
# CPU。本脚本按驱动支持的 CUDA 版本重装对应的 wheel。
#
# 版本对应（脚本自动选，-Index 可覆盖）：
#   cu118 (11.8) / cu121 (12.1) / cu124 (12.4) / cu126 (12.6) / cu128 (12.8) / cu129 (12.9)
# **RTX 50 系是 Blackwell（compute capability 12.0），必须 cu128 及以上**：
# cu126/cu124 的 wheel 不含 sm_120 kernel，cuda_available 会是 True 但一开训练就报
# "no kernel image is available for execution on the device"。
#
# 参数：
#   -Index <cuXXX>  跳过自动选择，强制指定 CUDA wheel 版本
#   -Yes            不询问，直接执行
#   -DryRun         只打印将要执行的命令，不做任何改动
#   -SkipMonitor    不装 nvidia-ml-py（训练仍用 GPU，但训练详情页没有 GPU 利用率面板）

param(
    [string]$Index = "",
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$SkipMonitor
)

# uv / nvidia-smi 会把进度与警告写到 stderr；ErrorActionPreference=Stop 会把它们
# 当成异常中断脚本，因此这里显式检查退出码。
$ErrorActionPreference = "Continue"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Stop-Script([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    exit 1
}

function Invoke-Uv {
    param([string[]]$UvArgs)
    $display = "uv " + ($UvArgs -join " ")
    if ($DryRun) {
        Write-Host "  [dry-run] $display" -ForegroundColor DarkGray
        return
    }
    Write-Host "  $display" -ForegroundColor DarkGray
    & uv @UvArgs
    if ($LASTEXITCODE -ne 0) {
        Stop-Script "uv 执行失败（退出码 $LASTEXITCODE）。修好网络/磁盘后重跑本脚本；torch 缺失时脚本会重新装。"
    }
}

function Get-NvidiaSmi {
    $candidates = @(
        (Join-Path $env:SystemRoot "System32\nvidia-smi.exe"),
        (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$venv = Join-Path $backend ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Stop-Script "未找到虚拟环境：$python`n请先执行： cd backend; uv venv --python 3.9 .venv; `$env:VIRTUAL_ENV='$venv'; uv pip install -r requirements.txt"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if ($DryRun) {
        Write-Host "[warn] PATH 里没有 uv，dry-run 仍会打印命令但无法执行。" -ForegroundColor Yellow
    } else {
        Stop-Script "PATH 里没有 uv，无法安装。请先装 uv（https://docs.astral.sh/uv/），或手动执行脚本打印的命令。"
    }
}

# ---------- 1. 探测 GPU 与驱动 ----------
Write-Step "探测 NVIDIA GPU 与驱动"

$smi = Get-NvidiaSmi
if (-not $smi) {
    Stop-Script @"
没找到 nvidia-smi，无法确认 NVIDIA 驱动。

CUDA 只认 NVIDIA 显卡：AMD / Intel 显卡没有对应路径，只能继续用 CPU 训练。
如果本机确实插着 N 卡，请先安装 NVIDIA 驱动后再重跑本脚本。
"@
}

# 查询参数必须整体加引号：PowerShell 参数模式会把逗号当数组分隔符，
# 不加引号时 nvidia-smi 收到的是拆开的多个参数，查询直接失败。
$gpuLines = & $smi "--query-gpu=name,compute_cap,driver_version,memory.total" "--format=csv,noheader" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $gpuLines) {
    # 老驱动没有 compute_cap 字段
    $gpuLines = & $smi "--query-gpu=name,driver_version,memory.total" "--format=csv,noheader" 2>$null
}
if ($LASTEXITCODE -ne 0 -or -not $gpuLines) {
    Stop-Script "nvidia-smi 查询失败（退出码 $LASTEXITCODE）。请先确认驱动正常。"
}
$gpus = @()
foreach ($line in $gpuLines) {
    $parts = @($line -split "," | ForEach-Object { $_.Trim() })
    $name = $parts[0]
    $cap = ""
    $driver = ""
    $mem = ""
    if ($parts.Count -ge 4) {
        # nvidia-smi 带 compute_cap 字段
        $cap = $parts[1]
        $driver = $parts[2]
        $mem = $parts[3]
    } else {
        $driver = $parts[1]
        $mem = $parts[2]
    }
    $gpus += [pscustomobject]@{ Name = $name; ComputeCap = $cap; Driver = $driver; Memory = $mem }
}

if (-not $gpus[0].Name) {
    Stop-Script "解析 nvidia-smi 输出失败：$gpuLines`n请手动执行 nvidia-smi --query-gpu=name --format=csv,noheader 确认驱动状态。"
}

$full = (& $smi 2>$null) -join "`n"
$m = [regex]::Match($full, "CUDA Version:\s*([0-9]+\.[0-9]+)")
$driverCuda = if ($m.Success) { [version]$m.Groups[1].Value } else { $null }

foreach ($g in $gpus) {
    Write-Host ("  {0} | 显存 {1} | 驱动 {2}" -f $g.Name, $g.Memory, $g.Driver)
}
if ($driverCuda) {
    Write-Host "  驱动支持的 CUDA 运行时：$driverCuda"
} else {
    Write-Host "  驱动未报告 CUDA Version（驱动过旧？）" -ForegroundColor Yellow
}

# ---------- 2. 选择 CUDA wheel ----------
Write-Step "选择 CUDA wheel"

$catalog = [ordered]@{
    "cu129" = [version]"12.9"
    "cu128" = [version]"12.8"
    "cu126" = [version]"12.6"
    "cu124" = [version]"12.4"
    "cu121" = [version]"12.1"
    "cu118" = [version]"11.8"
}

if ($Index) {
    if (-not $catalog.Contains($Index)) {
        Stop-Script "-Index 取值无效：$Index（可选 $($catalog.Keys -join ' / ')）"
    }
    $picked = $Index
    Write-Host "  使用指定的 $picked" -ForegroundColor Yellow
} else {
    if (-not $driverCuda) {
        Stop-Script "驱动没报告 CUDA 版本，无法自动选择。请用 -Index cu128 之类的参数手动指定。"
    }
    $picked = $null
    foreach ($k in $catalog.Keys) {
        if ($driverCuda -ge $catalog[$k]) { $picked = $k; break }
    }
    if (-not $picked) {
        Stop-Script "驱动只支持到 CUDA $driverCuda，低于最低候选 11.8。请升级 NVIDIA 驱动。"
    }
    Write-Host "  自动选择 $picked（CUDA $($catalog[$picked])）"
}

$capMajor = 0
if ($gpus[0].ComputeCap -match "^(\d+)") { $capMajor = [int]$Matches[1] }
if ($capMajor -ge 12 -and $catalog[$picked] -lt [version]"12.8") {
    Stop-Script @"
检测到 Blackwell 架构（compute capability $($gpus[0].ComputeCap)），需要 CUDA 12.8 及以上的 wheel，
但驱动只报告 CUDA $driverCuda，最高只能选到 $picked。

$picked 的 wheel 不含 sm_120 kernel —— 装上去 cuda_available 会是 True，
一开训练就报 "no kernel image is available for execution on the device"。

请升级 NVIDIA 驱动（需支持 CUDA 12.8+）后重跑本脚本。
"@
}

# ---------- 3. 现状 ----------
Write-Step "当前环境"

& $python -c @"
try:
    import torch
    print('  torch          :', torch.__version__)
    print('  cuda 运行时     :', torch.version.cuda)
    print('  cuda_available :', torch.cuda.is_available())
except Exception as exc:
    print('  torch 不可用    :', type(exc).__name__, exc)
"@

if ($env:YOLO_STUDIO_PYTHON) {
    Write-Host "  注意：YOLO_STUDIO_PYTHON=$env:YOLO_STUDIO_PYTHON，训练用的是那个解释器；本脚本装的是 $venv" -ForegroundColor Yellow
}

# ---------- 4. 执行 ----------
Write-Step "安装计划"

$installArgs = @("pip", "install", "--python", $python, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/$picked")
$monitorArgs = @("pip", "install", "--python", $python, "nvidia-ml-py")

Write-Host "  1) uv pip uninstall --python <venv> torch torchvision   （先卸掉 CPU 版）"
Write-Host "  2) uv $($installArgs -join ' ')"
if (-not $SkipMonitor) {
    Write-Host "  3) uv $($monitorArgs -join ' ')   （GPU 利用率面板）"
}

if (-not $Yes -and -not $DryRun) {
    Write-Host ""
    $answer = Read-Host "继续？(y/N)"
    if ($answer -notmatch "^(y|yes)$") {
        Write-Host "已取消。" -ForegroundColor Yellow
        exit 0
    }
}

Write-Step "执行安装"

Invoke-Uv @("pip", "uninstall", "--python", $python, "torch", "torchvision")
Invoke-Uv $installArgs
if (-not $SkipMonitor) { Invoke-Uv $monitorArgs }

if ($DryRun) {
    Write-Host ""
    Write-Host "dry-run 结束：没有做任何改动。" -ForegroundColor Yellow
    exit 0
}

# ---------- 5. 验证 ----------
Write-Step "验证"

& $python -c @"
import torch
print('  torch          :', torch.__version__)
print('  cuda 运行时     :', torch.version.cuda)
print('  cuda_available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  设备           :', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    try:
        x = torch.randn(1024, 1024, device='cuda')
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print('  cuda 矩阵乘法   : ok')
    except Exception as exc:
        print('  cuda 矩阵乘法   : 失败 ->', type(exc).__name__, exc)
try:
    import pynvml
    pynvml.nvmlInit()
    print('  pynvml         : ok，', pynvml.nvmlDeviceGetCount(), '个设备')
except Exception as exc:
    print('  pynvml         : 不可用 ->', type(exc).__name__, exc)
"@

Write-Host ""
Write-Host "下一步：重启后端（scripts\start-backend.ps1）——换了 torch 这种二进制必须重启进程，--reload 不够。" -ForegroundColor Green
Write-Host "  自查 GET /api/system/env     → cuda_available=true、devices 列出显卡、train_device=cuda:0"
Write-Host "  自查 GET /api/train/backends → monitor.gpu_monitor=true、defaults.device=cuda:0"
Write-Host "  设备不用配置：YOLO_STUDIO_DEVICE 默认 auto，有 CUDA 就用 cuda:0。"
Write-Host ""
Write-Host "注意：以后不要跑 uv pip install -r requirements.txt --reinstall，那会把 torch 换回 CPU 版。" -ForegroundColor Yellow
