import { Alert, Card, Col, Empty, Progress, Row, Statistic, Tag, Typography } from 'antd'
import type { ResourceSample } from '../types'
import { formatBytes } from '../utils'

const { Text } = Typography

interface Props {
  sample: ResourceSample | null
  monitor?: { cpu_monitor: boolean; gpu_monitor: boolean; gpu_count: number; gpu_error: string } | null
}

/** CPU / GPU 资源面板。无 NVIDIA 设备时只显示 CPU，并把原因说清楚。 */
export default function ResourcePanel({ sample, monitor }: Props) {
  const cpu = sample?.cpu
  const gpu = sample?.gpu
  const proc = cpu?.process

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} lg={gpu?.available ? 12 : 24}>
        <Card title="CPU / 内存" size="small">
          {cpu?.available ? (
            <>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic title="整机 CPU" value={cpu.percent ?? 0} suffix="%" precision={1} />
                  <Progress
                    percent={Math.round(cpu.percent ?? 0)}
                    size="small"
                    status="active"
                    showInfo={false}
                  />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {cpu.count ?? '-'} 逻辑核心
                  </Text>
                </Col>
                <Col span={12}>
                  <Statistic
                    title="整机内存"
                    value={cpu.mem_percent ?? 0}
                    suffix="%"
                    precision={1}
                  />
                  <Progress
                    percent={Math.round(cpu.mem_percent ?? 0)}
                    size="small"
                    showInfo={false}
                  />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {formatBytes(cpu.mem_used)} / {formatBytes(cpu.mem_total)}
                  </Text>
                </Col>
              </Row>

              {proc && (
                <div style={{ marginTop: 12 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    训练进程 pid={proc.pid}
                  </Text>
                  <Row gutter={16} style={{ marginTop: 4 }}>
                    <Col span={8}>
                      <Statistic
                        title="进程 CPU"
                        value={proc.cpu_percent ?? 0}
                        suffix="%"
                        precision={1}
                        valueStyle={{ fontSize: 16 }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="进程内存"
                        value={formatBytes(proc.mem_rss)}
                        valueStyle={{ fontSize: 16 }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="线程 / 子进程"
                        value={`${proc.threads ?? '-'} / ${proc.children ?? '-'}`}
                        valueStyle={{ fontSize: 16 }}
                      />
                    </Col>
                  </Row>
                </div>
              )}
            </>
          ) : (
            <Empty
              description={
                monitor?.cpu_monitor === false
                  ? 'psutil 未安装，无法采集 CPU 资源'
                  : '暂无资源采样（任务启动后开始采集）'
              }
            />
          )}
        </Card>
      </Col>

      <Col xs={24} lg={12}>
        <Card
          title={
            <span>
              GPU{' '}
              {gpu?.available ? (
                <Tag color="green">{gpu.devices.length} 卡</Tag>
              ) : (
                <Tag color="orange">不可用</Tag>
              )}
            </span>
          }
          size="small"
        >
          {gpu?.available && gpu.devices.length > 0 ? (
            gpu.devices.map((d) => (
              <div key={d.index} style={{ marginBottom: 12 }}>
                <Text strong style={{ fontSize: 13 }}>
                  [{d.index}] {d.name ?? `GPU ${d.index}`}
                </Text>
                <Row gutter={16} style={{ marginTop: 4 }}>
                  <Col span={8}>
                    <Statistic
                      title="利用率"
                      value={d.utilization ?? 0}
                      suffix="%"
                      valueStyle={{ fontSize: 16 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="显存"
                      value={d.mem_percent ?? 0}
                      suffix="%"
                      precision={1}
                      valueStyle={{ fontSize: 16 }}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {formatBytes(d.mem_used)} / {formatBytes(d.mem_total)}
                    </Text>
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="温度"
                      value={d.temperature ?? '-'}
                      suffix="°C"
                      valueStyle={{ fontSize: 16 }}
                    />
                  </Col>
                </Row>
              </div>
            ))
          ) : (
            <Alert
              type="info"
              showIcon
              message="没有可用的 GPU 监控"
              description={
                <div>
                  <div>CPU 面板不受影响，训练仍在正常进行。</div>
                  {gpu?.error ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      原因：{gpu.error}
                    </Text>
                  ) : monitor?.gpu_error ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      原因：{monitor.gpu_error}
                    </Text>
                  ) : null}
                </div>
              }
            />
          )}
        </Card>
      </Col>
    </Row>
  )
}
