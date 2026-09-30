# 启动后端（FastAPI，开发模式，热重载）
# 用法： powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "未找到虚拟环境：$python`n请先执行： cd backend; uv venv --python 3.9 .venv; `$env:VIRTUAL_ENV='$backend\.venv'; uv pip install -r requirements.txt"
    exit 1
}

Set-Location $backend
Write-Host "[backend] http://127.0.0.1:8010  (docs: /docs)" -ForegroundColor Cyan
& $python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
