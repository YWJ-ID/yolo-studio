import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * 开发服务器允许的主机名（Vite 5.4.12+ 的主机校验，防 DNS 重绑定）。
 *
 * 默认只放行 localhost 与 IP 地址；用**主机名 / 域名**访问（如 http://my-pc:5173、
 * 隧道/反向代理域名）会被拦下并提示：
 *   Blocked request. This host ("my-pc") is not allowed.
 *
 * 放行方式（逗号分隔）：
 *   $env:VITE_ALLOWED_HOSTS = "my-pc,my-pc.lan,dev.example.com"
 * 特殊值 true / all / * 表示放行任意主机——会关闭该防护，仅限可信的本机/内网。
 * 也可写进 frontend/.env.local（loadEnv 会读取）。
 */
function readAllowedHosts(mode: string): string[] | true | undefined {
  const env = loadEnv(mode, '.', '')
  const raw = (env.VITE_ALLOWED_HOSTS ?? '').trim()
  if (!raw) return undefined
  if (['true', '1', 'all', '*'].includes(raw.toLowerCase())) return true
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

/**
 * API 超时（毫秒），默认 5 分钟。与前端 axios 读同一个 VITE_API_TIMEOUT，
 * 避免出现「浏览器还在等、代理已经断开」的不一致。
 */
function readTimeoutMs(mode: string): number {
  const raw = (loadEnv(mode, '.', '').VITE_API_TIMEOUT ?? '').trim()
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : 300000
}

// 前后端分离：开发时前端 5173，API 请求代理到后端 8010。
// 要让局域网访问前端开发服务：npm run dev -- --host 0.0.0.0，
// 并在后端设置 YOLO_STUDIO_CORS_ORIGINS 放行访问来源。
export default defineConfig(({ mode }) => {
  const timeoutMs = readTimeoutMs(mode)
  return {
    plugins: [react()],
    server: {
      port: 5173,
      allowedHosts: readAllowedHosts(mode),
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8010',
          changeOrigin: true,
          // 训练/评估/导出的实时推送走 WebSocket，代理必须显式开启
          ws: true,
          // 数据管道的 scan/clean/split/export 是同步接口，可能跑好几分钟
          timeout: timeoutMs,
          proxyTimeout: timeoutMs,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      // 不强制 manualChunks：交给 Rollup 按「页面懒加载 + 第三方依赖」自动分包，
      // antd 的按需组件各自成块、只在该页面用到时才加载。echarts 已在
      // src/components/echarts.tsx 里按需注册，体积由 ~1.0MB 降到 ~0.57MB。
    },
  }
})
