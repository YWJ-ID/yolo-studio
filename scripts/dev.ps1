# 一键启动前后端开发环境（两个独立窗口）
# 用法： powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
#
# 前后端分离：后端 8000，前端 5173（通过 Vite 代理调用 /api）

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "启动 YOLO Studio 开发环境…" -ForegroundColor Green

Start-Process powershell -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "scripts\start-backend.ps1")
)

Start-Sleep -Seconds 2

Start-Process powershell -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "scripts\start-frontend.ps1")
)

Write-Host ""
Write-Host "  后端 API   http://127.0.0.1:8010/docs" -ForegroundColor Cyan
Write-Host "  前端界面   http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "关闭对应的 PowerShell 窗口即可停止服务。" -ForegroundColor DarkGray
