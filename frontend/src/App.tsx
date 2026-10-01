import { lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'

// 路由级懒加载：把 AntD + ECharts 的主包拆成按页加载的 chunk，
// 首屏只加载总览所需的代码（MainLayout 里的 Suspense 负责过渡）。
const Dashboard = lazy(() => import('./pages/Dashboard'))
const DatasetImport = lazy(() => import('./pages/DatasetImport'))
const DatasetBrowser = lazy(() => import('./pages/DatasetBrowser'))
const DatasetVersions = lazy(() => import('./pages/DatasetVersions'))
const QualityReport = lazy(() => import('./pages/QualityReport'))
const TrainJobs = lazy(() => import('./pages/TrainJobs'))
const TrainNew = lazy(() => import('./pages/TrainNew'))
const TrainDetail = lazy(() => import('./pages/TrainDetail'))
const ModelLibrary = lazy(() => import('./pages/ModelLibrary'))
const ModelCompare = lazy(() => import('./pages/ModelCompare'))
const ModelDetail = lazy(() => import('./pages/ModelDetail'))
const VerifyCenter = lazy(() => import('./pages/VerifyCenter'))
const Settings = lazy(() => import('./pages/Settings'))

export default function App() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/" element={<Dashboard />} />

        <Route path="/data/import" element={<DatasetImport />} />
        <Route path="/data/browser" element={<DatasetBrowser />} />
        <Route path="/data/versions" element={<DatasetVersions />} />
        <Route path="/data/quality" element={<QualityReport />} />

        <Route path="/train" element={<TrainJobs />} />
        <Route path="/train/new" element={<TrainNew />} />
        <Route path="/train/:jobId" element={<TrainDetail />} />

        <Route path="/models" element={<ModelLibrary />} />
        <Route path="/models/compare" element={<ModelCompare />} />
        <Route path="/models/:modelId" element={<ModelDetail />} />

        <Route path="/verify" element={<VerifyCenter />} />

        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
