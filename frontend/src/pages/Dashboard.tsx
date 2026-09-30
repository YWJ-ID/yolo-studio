import { useEffect, useState } from 'react'
import { Alert, Card, Col, Descriptions, Row, Spin, Statistic, Table, Tag, Typography } from 'antd'
import { api } from '../api/client'
import type { AdapterInfo, EnvResponse, HealthResponse } from '../types'

const { Text, Paragraph } = Typography

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [env, setEnv] = useState<EnvResponse | null>(null)
  const [adapters, setAdapters] = useState<AdapterInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.health(), api.env(), api.adapters()])
      .then(([h, e, a]) => {
        setHealth(h)
        setEnv(e)
        setAdapters(a)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spin style={{ display: 'block', marginTop: 80 }} />

  return (
    <div className="page">
      {error && (
        <Alert
          type="error"
          showIcon
          message="后端连接失败"
          description={`${error}　—— 请确认后端已启动：powershell -ExecutionPolicy Bypass -File scripts\\start-backend.ps1`}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="运行环境" size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="服务"> {health?.app ?? '-'} v{health?.version ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="Python">{env?.python ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="ultralytics">
                {env?.ultralytics ? <Tag color="green">{env.ultralytics}</Tag> : <Tag>未安装</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="PyTorch">{env?.torch ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="CUDA">
                {env?.cuda_available ? (
                  <Tag color="green">可用（{env.cuda_device_count} 卡）</Tag>
                ) : (
                  <Tag color="orange">不可用，训练将以 CPU 运行</Tag>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="存储目录">
                <Text copyable className="mono">{health?.storage_dir ?? '-'}</Text>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="已接入的数据格式" size="small">
            <Table
              size="small"
              rowKey="name"
              pagination={false}
              dataSource={adapters}
              columns={[
                { title: '标识', dataIndex: 'name', width: 100 },
                { title: '说明', dataIndex: 'display_name' },
              ]}
            />
            <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              当前骨架仅实现 YOLO 格式；COCO / VOC / LabelMe / OpenLABEL(DMD) 见任务 M1-03 ~ M1-06。
            </Paragraph>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="路线图" size="small">
            <Row gutter={16}>
              <Col xs={12} md={4}>
                <Statistic title="M0 骨架" value="进行中" valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col xs={12} md={4}>
                <Statistic title="M1 数据模块" value="1/12" valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col xs={12} md={4}>
                <Statistic title="M2 训练模块" value="0/7" valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col xs={12} md={4}>
                <Statistic title="M3 评估/模型库" value="0/4" valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col xs={12} md={4}>
                <Statistic title="M4 部署导出" value="0/3" valueStyle={{ fontSize: 16 }} />
              </Col>
              <Col xs={12} md={4}>
                <Statistic title="M5 前端界面" value="2/8" valueStyle={{ fontSize: 16 }} />
              </Col>
            </Row>
            <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              完整任务清单见项目根目录 <Text code>PROGRESS.md</Text>。
            </Paragraph>
          </Card>
        </Col>
      </Row>
    </div>
  )
}
