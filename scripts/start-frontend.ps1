# 启动前端（Vite，开发模式，热更新）
# 用法： powershell -ExecutionPolicy Bypass -File scripts\start-frontend.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "frontend"

Set-Location $frontend
if (-not (Test-Path "node_modules")) {
    Write-Host "[frontend] 首次运行，安装依赖…" -ForegroundColor Yellow
    npm install
}

Write-Host "[frontend] http://localhost:5173" -ForegroundColor Cyan
npm run dev
