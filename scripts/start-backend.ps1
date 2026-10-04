# 启动后端（FastAPI，开发模式，热重载）
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1 -BindHost 0.0.0.0 -Port 8010
#
# -BindHost 默认 127.0.0.1（仅本机）；要让别的机器访问改成 0.0.0.0（并放行防火墙端口）。

param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "未找到虚拟环境：$python`n请先执行： cd backend; uv venv --python 3.9 .venv; `$env:VIRTUAL_ENV='$backend\.venv'; uv pip install -r requirements.txt"
    exit 1
}

Set-Location $backend
Write-Host "[backend] http://${BindHost}:${Port}  (docs: /docs)" -ForegroundColor Cyan
if ($BindHost -ne "127.0.0.1" -and $BindHost -ne "localhost") {
    Write-Host "[backend] 已监听所有网卡：请确认防火墙已放行该端口，且已按需设置 YOLO_STUDIO_ALLOWED_ROOTS / YOLO_STUDIO_CORS_ORIGINS" -ForegroundColor Yellow
}
& $python -m uvicorn app.main:app --reload --host $BindHost --port $Port
