import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 前后端分离：开发时前端 5173，API 请求代理到后端 8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
        // 训练/评估/导出的实时推送走 WebSocket，代理必须显式开启
        ws: true,
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
})
