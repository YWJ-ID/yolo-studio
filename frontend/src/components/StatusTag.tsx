import { Tag } from 'antd'
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  PauseCircleOutlined,
  SyncOutlined,
  WarningOutlined,
} from '@ant-design/icons'

/** 训练 / 评估 / 导出共用同一套状态取值（后端 job.py 保持一致），因此映射也共用。 */
const STATUS_META: Record<
  string,
  { color: string; label: string; icon?: React.ReactNode }
> = {
  pending: { color: 'default', label: '排队中', icon: <ClockCircleOutlined /> },
  running: { color: 'processing', label: '进行中', icon: <SyncOutlined spin /> },
  stopping: { color: 'warning', label: '停止中', icon: <LoadingOutlined /> },
  stopped: { color: 'default', label: '已停止', icon: <PauseCircleOutlined /> },
  finished: { color: 'success', label: '已完成', icon: <CheckCircleOutlined /> },
  failed: { color: 'error', label: '失败', icon: <CloseCircleOutlined /> },
  interrupted: { color: 'warning', label: '已中断', icon: <WarningOutlined /> },
}

export const ACTIVE_STATUSES = ['pending', 'running', 'stopping']

export function isActiveStatus(status: string): boolean {
  return ACTIVE_STATUSES.includes(status)
}

export default function StatusTag({
  status,
  label,
}: {
  status: string
  label?: string
}) {
  const meta = STATUS_META[status] ?? { color: 'default', label: status }
  return (
    <Tag color={meta.color} icon={meta.icon}>
      {label || meta.label}
    </Tag>
  )
}
