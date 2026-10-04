import { useEffect, useState } from 'react'
import {
  Alert,
  Card,
  Col,
  Descriptions,
  Row,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd'
import { api } from '../api/client'
import type { EnvResponse, SystemConfig, TrainBackendsResponse } from '../types'

const { Text, Paragraph } = Typography

const PATH_LABELS: Record<string, string> = {
  storage_dir: '存储根目录',
  datasets_dir: '数据集版本',
  runs_dir: '训练产物',
  weights_dir: '预训练权重缓存',
  evals_dir: '评估产物',
  models_dir: '模型库',
  deploys_dir: '导出产物',
  uploads_dir: '上传/导入原始数据',
  reports_dir: '质量报告',
  db_path: 'SQLite 数据库（保留字段，当前未使用）',
}

/** 设置页（M0-04 的前端呈现）：只读展示生效配置与运行环境，并说明如何覆盖。 */
export default function Settings() {
  const [config, setConfig] = useState<SystemConfig | null>(null)
  const [env, setEnv] = useState<EnvResponse | null>(null)
  const [backends, setBackends] = useState<TrainBackendsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api.systemConfig(), api.env(), api.trainBackends()])
      .then(([c, e, b]) => {
        setConfig(c)
        setEnv(e)
        setBackends(b)
      })
      .catch((err) => setError(err.message ?? '读取配置失败'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spin style={{ display: 'block', marginTop: 80 }} />

  if (error && !config) {
    return (
      <div className="page">
        <Alert type="error" showIcon message="读取配置失败" description={error} />
      </div>
    )
  }

  return (
    <div className="page">
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="配置在后端启动时从环境变量读取，运行期不可修改"
        description={
          <span>
            下表里的每个路径都可用对应的 <Text code>YOLO_STUDIO_*</Text> 环境变量覆盖；
            修改后需要重启后端进程才会生效。没有设置环境变量时使用默认值。
          </span>
        }
      />

      <Card title="运行环境" size="small">
        <Descriptions column={{ xs: 1, md: 2 }} size="small">
          <Descriptions.Item label="服务">
            {config?.app} v{config?.version}
          </Descriptions.Item>
          <Descriptions.Item label="API 端口">{config?.api_port}</Descriptions.Item>
          <Descriptions.Item label="平台">
            <Text className="mono" style={{ fontSize: 11 }}>
              {config?.platform ?? env?.platform ?? '-'}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="Python">{env?.python ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="后端解释器">
            <Text copyable className="mono" style={{ fontSize: 11 }}>
              {config?.python_executable ?? '-'}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="训练解释器">
            <Text copyable className="mono" style={{ fontSize: 11 }}>
              {config?.training_python ?? '-'}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="ultralytics">
            {env?.ultralytics ? <Tag color="green">{env.ultralytics}</Tag> : <Tag>未安装</Tag>}
          </Descriptions.Item>
          <Descriptions.Item label="PyTorch">{env?.torch ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="CUDA">
            {env?.cuda_available ? (
              <Tag color="green">可用（{env.cuda_device_count} 卡）</Tag>
            ) : (
              <Tag color="orange">不可用，训练以 CPU 运行</Tag>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="默认训练设备">
            <Tag color="blue">{config?.train_device ?? '-'}</Tag>
            <Text type="secondary" style={{ fontSize: 11, marginLeft: 6 }}>
              由 YOLO_STUDIO_DEVICE 控制
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="资源监控">
            <Space size={4}>
              <Tag color={backends?.monitor.cpu_monitor ? 'green' : 'orange'}>CPU</Tag>
              <Tag color={backends?.monitor.gpu_monitor ? 'green' : 'orange'}>
                GPU{backends?.monitor.gpu_count ? `×${backends.monitor.gpu_count}` : ''}
              </Tag>
            </Space>
            {backends?.monitor.gpu_error && !backends.monitor.gpu_monitor && (
              <Text type="secondary" style={{ fontSize: 11, marginLeft: 6 }}>
                {backends.monitor.gpu_error}
              </Text>
            )}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="存储路径与配置" size="small" style={{ marginTop: 16 }}>
        <Table
          size="small"
          rowKey="key"
          pagination={false}
          dataSource={config?.paths ?? []}
          columns={[
            {
              title: '用途',
              dataIndex: 'key',
              width: 160,
              render: (k: string) => PATH_LABELS[k] ?? k,
            },
            {
              title: '环境变量',
              dataIndex: 'env',
              width: 210,
              render: (v: string) => (
                <Text code style={{ fontSize: 12 }}>
                  {v}
                </Text>
              ),
            },
            {
              title: '当前值',
              dataIndex: 'value',
              render: (v: string) => (
                <Text copyable className="mono" style={{ fontSize: 11 }}>
                  {v}
                </Text>
              ),
            },
            {
              title: '状态',
              dataIndex: 'exists',
              width: 90,
              render: (exists: boolean) =>
                exists ? <Tag color="success">已存在</Tag> : <Tag>尚未创建</Tag>,
            },
          ]}
        />
        <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
          「尚未创建」不是错误：目录会在首次写入时自动创建（<Text code>ensure_dirs()</Text>）。
        </Paragraph>
      </Card>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="可选依赖与能力" size="small">
            <Table
              size="small"
              rowKey="name"
              pagination={false}
              dataSource={config?.optional_dependencies ?? []}
              columns={[
                {
                  title: '包',
                  dataIndex: 'name',
                  width: 120,
                  render: (name: string, r) => (
                    <Space size={6}>
                      <Text code style={{ fontSize: 12 }}>
                        {name}
                      </Text>
                      {r.installed ? <Tag color="success">已装</Tag> : <Tag color="default">缺失</Tag>}
                    </Space>
                  ),
                },
                { title: '用途', dataIndex: 'purpose' },
                {
                  title: '安装',
                  dataIndex: 'install',
                  width: 220,
                  render: (v: string) => (
                    <Text className="mono" style={{ fontSize: 11 }} copyable={{ text: v }}>
                      {v}
                    </Text>
                  ),
                },
              ]}
            />
            <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
              这些是可选依赖：缺了不影响基本功能，只是对应能力不可用。导出格式的可用性与此一致。
            </Paragraph>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="导出格式能力" size="small">
            <Table
              size="small"
              rowKey="name"
              pagination={false}
              dataSource={config?.export_formats ?? []}
              columns={[
                { title: '格式', dataIndex: 'label', width: 150 },
                {
                  title: '状态',
                  dataIndex: 'available',
                  width: 90,
                  render: (v: boolean) => (v ? <Tag color="green">可用</Tag> : <Tag color="orange">不可用</Tag>),
                },
                {
                  title: '说明',
                  render: (_, r) => (r.available ? <Text type="secondary">{r.note}</Text> : r.reason),
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <Card title="访问与安全" size="small" style={{ marginTop: 16 }}>
        {config?.files_read_unrestricted ? (
          <Alert
            type="warning"
            showIcon
            message="图片读取未设白名单（默认，仅适用于本机单机开发）"
            description={
              <span>
                当前 <Text code>/api/files/image</Text> 可读取本机任意图片。若要把服务暴露到局域网/公网，
                必须设置 <Text code>YOLO_STUDIO_ALLOWED_ROOTS</Text>（分号分隔的根目录白名单）后重启。
              </span>
            }
          />
        ) : (
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              图片读取白名单（YOLO_STUDIO_ALLOWED_ROOTS）：
            </Text>
            <ul style={{ marginBottom: 0, marginTop: 8, paddingLeft: 20 }}>
              {(config?.allowed_roots ?? []).map((r) => (
                <li key={r}>
                  <Text copyable className="mono" style={{ fontSize: 11 }}>
                    {r}
                  </Text>
                </li>
              ))}
            </ul>
          </div>
        )}
        <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0, fontSize: 12 }}>
          CORS 允许来源（YOLO_STUDIO_CORS_ORIGINS）：
          <Space size={4} wrap style={{ marginLeft: 4 }}>
            {(config?.cors_origins ?? []).map((o) => (
              <Tag key={o} className="mono">
                {o}
              </Tag>
            ))}
          </Space>
          <br />
          同源访问（后端托管前端，单端口 <Text code>scripts/start-server.ps1</Text>）不受 CORS 限制；
          开发模式经 Vite 访问时，若来源不在上面，会被浏览器拦下。
        </Paragraph>
        <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
          生产前端托管目录：<Text code>{config?.frontend_dist}</Text>{' '}
          {config?.frontend_dist_exists ? (
            <Tag color="success">已构建</Tag>
          ) : (
            <Tag color="orange">未构建（开发模式用 Vite 5173）</Tag>
          )}
        </Paragraph>
      </Card>
    </div>
  )
}
