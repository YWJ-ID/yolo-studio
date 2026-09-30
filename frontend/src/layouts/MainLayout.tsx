import { Suspense } from 'react'
import { Layout, Menu, Spin, Typography } from 'antd'
import {
  AppstoreOutlined,
  BarChartOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  DeploymentUnitOutlined,
  SettingOutlined,
  CloudUploadOutlined,
  PictureOutlined,
  HistoryOutlined,
} from '@ant-design/icons'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'

const { Sider, Header, Content } = Layout

const MENU_ITEMS = [
  { key: '/', icon: <AppstoreOutlined />, label: '总览' },
  {
    key: 'data',
    icon: <DatabaseOutlined />,
    label: '数据',
    children: [
      { key: '/data/import', icon: <CloudUploadOutlined />, label: '数据导入' },
      { key: '/data/browser', icon: <PictureOutlined />, label: '数据浏览' },
      { key: '/data/quality', icon: <BarChartOutlined />, label: '质量报告' },
      { key: '/data/versions', icon: <HistoryOutlined />, label: '数据集版本' },
    ],
  },
  { key: '/train', icon: <ExperimentOutlined />, label: '训练' },
  { key: '/models', icon: <DeploymentUnitOutlined />, label: '模型库' },
  { key: '/settings', icon: <SettingOutlined />, label: '设置' },
]

/** 把子页面（详情 / 新建 / 对比）归到所属的菜单项，保证侧边栏高亮正确。 */
function selectedMenuKey(pathname: string): string {
  if (pathname.startsWith('/train')) return '/train'
  if (pathname.startsWith('/models')) return '/models'
  if (pathname.startsWith('/data/')) return pathname
  return pathname
}

export default function MainLayout() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider theme="dark" width={216} breakpoint="lg" collapsedWidth={64}>
        <div className="logo">
          <span className="logo-mark">YS</span>
          <span className="logo-text">YOLO Studio</span>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedMenuKey(location.pathname)]}
          defaultOpenKeys={['data']}
          items={MENU_ITEMS}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>

      <Layout>
        <Header className="app-header">
          <Typography.Text strong>YOLO 训练全流程可视化工作台</Typography.Text>
        </Header>
        <Content className="app-content">
          <Suspense fallback={<Spin style={{ display: 'block', marginTop: 80 }} />}>
            <Outlet />
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  )
}
