import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 개발 서버에서 /api/* 요청을 FastAPI 백엔드로 전달 (CORS 문제 방지)
    proxy: {
      '/api': 'http://localhost:8001',
    },
  },
})
