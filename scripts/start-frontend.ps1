# 启动前端（Vite，开发模式，热更新）
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\start-frontend.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start-frontend.ps1 -BindHost 0.0.0.0
#
# -BindHost 留空 = Vite 默认（仅本机）；要让局域网访问传 0.0.0.0。
# 注意：Vite 代理把 /api 转发到本机后端 8010，所以后端要在同一台机器上运行。

param(
    [string]$BindHost = "",
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "frontend"

Set-Location $frontend
if (-not (Test-Path "node_modules")) {
    Write-Host "[frontend] 首次运行，安装依赖…" -ForegroundColor Yellow
    npm install
}

$viteArgs = @("--port", "$Port")
if ($BindHost) { $viteArgs += @("--host", $BindHost) }

$shown = if ($BindHost) { $BindHost } else { "localhost" }
Write-Host "[frontend] http://${shown}:${Port}" -ForegroundColor Cyan
if ($BindHost) {
    Write-Host "[frontend] 已暴露到网络：若用 IP 访问前端开发服务，需要在后端设置 YOLO_STUDIO_CORS_ORIGINS 放行该来源" -ForegroundColor Yellow
}
npm run dev -- $viteArgs
