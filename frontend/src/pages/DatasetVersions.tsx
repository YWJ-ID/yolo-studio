import { useEffect, useState } from 'react'
import { Button, Card, Descriptions, Empty, Modal, Space, Table, Tag, Typography, message } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { api } from '../api/client'
import type { DatasetVersion } from '../types'

const { Text, Paragraph } = Typography

const TASK_LABEL: Record<string, { text: string; color: string }> = {
  detection: { text: '目标检测', color: 'blue' },
  classification: { text: '图像分类', color: 'purple' },
}

/** 已生成的数据集版本列表 + 血缘查看。 */
export default function DatasetVersions() {
  const [versions, setVersions] = useState<DatasetVersion[]>([])
  const [loading, setLoading] = useState(false)
  const [detail, setDetail] = useState<Record<string, any> | null>(null)
  const [detailName, setDetailName] = useState('')

  const load = () => {
    setLoading(true)
    api
      .versions()
      .then(setVersions)
      .catch((err) => message.error(err.message ?? '读取失败'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const openDetail = async (name: string) => {
    try {
      setDetail(await api.version(name))
      setDetailName(name)
    } catch (err: any) {
      message.error(err.message ?? '读取血缘失败')
    }
  }

  return (
    <div className="page">
      <Card
        title="数据集版本"
        size="small"
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        }
      >
        <Paragraph type="secondary">
          每次「生成数据集」都会产出不可变的一个版本，并记录来源、划分参数与随机种子（血缘）。
          训练时据此可反查模型是在哪一版数据上训出来的。
        </Paragraph>

        {versions.length === 0 && !loading ? (
          <Empty description="还没有生成过数据集，请到「数据导入」页面生成" />
        ) : (
          <Table
            size="small"
            rowKey="path"
            loading={loading}
            dataSource={versions}
            pagination={false}
            columns={[
              {
                title: '名称',
                dataIndex: 'name',
                render: (v, r) => (
                  <Space direction="vertical" size={0}>
                    <Space size={4}>
                      <Text strong>{v}</Text>
                      {r.prelabel && (
                        <Tag color="orange" title="由模型预测生成，必须人工复核后才能作为训练数据">
                          伪标签
                        </Tag>
                      )}
                    </Space>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {r.path}
                    </Text>
                  </Space>
                ),
              },
              {
                title: '任务',
                dataIndex: 'task',
                width: 110,
                render: (t: string) => {
                  const info = TASK_LABEL[t]
                  return info ? <Tag color={info.color}>{info.text}</Tag> : <Tag>{t}</Tag>
                },
              },
              {
                title: '子集图像数',
                dataIndex: 'images',
                width: 260,
                render: (images: Record<string, number>) => (
                  <Space>
                    {['train', 'val', 'test'].map((k) => (
                      <Tag key={k} color={(images?.[k] ?? 0) === 0 ? 'red' : undefined}>
                        {k}: {images?.[k] ?? 0}
                      </Tag>
                    ))}
                  </Space>
                ),
              },
              {
                title: '类别数',
                dataIndex: 'classes',
                width: 90,
                render: (c: string[]) => c?.length ?? 0,
              },
              { title: '生成时间', dataIndex: 'created_at', width: 180 },
              {
                title: '操作',
                width: 100,
                render: (_, r) => (
                  <Button size="small" type="link" onClick={() => openDetail(r.name)}>
                    查看血缘
                  </Button>
                ),
              },
            ]}
          />
        )}
      </Card>

      <Modal
        open={!!detail}
        title={`血缘：${detailName}`}
        footer={null}
        width={760}
        onCancel={() => setDetail(null)}
      >
        {detail && <LineageDetail card={detail} />}
      </Modal>
    </div>
  )
}

function LineageDetail({ card }: { card: Record<string, any> }) {
  const split = card.split_report ?? {}
  const exportCfg = card.export ?? {}

  return (
    <Descriptions column={1} size="small" bordered>
      <Descriptions.Item label="任务">
        <Tag color={TASK_LABEL[card.task]?.color}>{TASK_LABEL[card.task]?.text ?? card.task}</Tag>
      </Descriptions.Item>
      <Descriptions.Item label="生成时间">{card.created_at}</Descriptions.Item>
      <Descriptions.Item label="类别">
        {(card.classes ?? []).map((c: string, i: number) => (
          <Tag key={c}>
            {i}: {c}
          </Tag>
        ))}
      </Descriptions.Item>
      <Descriptions.Item label="来源">
        {(card.sources ?? []).map((s: any, i: number) => (
          <div key={i} className="mono">
            [{s.format}] {s.root}
          </div>
        ))}
      </Descriptions.Item>
      {card.prelabel && (
        <Descriptions.Item label="预标注血缘">
          <div>
            <Tag color="orange">伪标签</Tag>
            权重 <Text className="mono">{card.prelabel.weights}</Text>
            ，conf {card.prelabel.conf}，iou {card.prelabel.iou}
            {Array.isArray(card.prelabel.classes) && card.prelabel.classes.length > 0 && (
              <>，模型类别 {(card.prelabel.classes as string[]).join('、')}</>
            )}
          </div>
          <Text type="danger">{card.prelabel.disclaimer}</Text>
        </Descriptions.Item>
      )}
      <Descriptions.Item label="划分参数">
        {split.warnings ? (
          <div>
            <div>
              单元 {split.units_total}（分组 {split.units_grouped}），已有划分{' '}
              {split.images_already_split}，本次分配 {split.images_assigned}
            </div>
            <div>
              子集: {JSON.stringify(split.split_images ?? {})}
            </div>
          </div>
        ) : (
          <Text type="secondary">未记录（导出时未执行划分）</Text>
        )}
      </Descriptions.Item>
      <Descriptions.Item label="导出参数">
        <div className="mono">
          落盘={exportCfg.file_mode} 命名={exportCfg.name_style} 裁剪越界框=
          {String(exportCfg.clamp_boxes)}
        </div>
      </Descriptions.Item>
      <Descriptions.Item label="统计">
        导出图 {JSON.stringify(card.images_exported ?? {})}，框 {card.boxes_exported}，
        跳过 {card.images_skipped}，裁剪框 {card.clamped_boxes}
      </Descriptions.Item>
      {(card.warnings ?? []).length > 0 && (
        <Descriptions.Item label="警告">
          <ul style={{ marginBottom: 0, paddingLeft: 18 }}>
            {(card.warnings as string[]).slice(0, 10).map((w, i) => (
              <li key={i}>
                <Text type="secondary">{w}</Text>
              </li>
            ))}
          </ul>
        </Descriptions.Item>
      )}
    </Descriptions>
  )
}
