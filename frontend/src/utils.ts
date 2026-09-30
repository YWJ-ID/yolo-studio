/** 通用格式化工具（训练 / 评估 / 导出页面共用）。 */

export function formatBytes(bytes?: number | null, digits = 1): string {
  if (bytes == null || !isFinite(bytes)) return '-'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i += 1
  }
  return `${value.toFixed(digits)} ${units[i]}`
}

export function formatDuration(sec?: number | null): string {
  if (sec == null || !isFinite(sec)) return '-'
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  if (m < 60) return `${m} 分 ${s} 秒`
  const h = Math.floor(m / 60)
  return `${h} 时 ${m % 60} 分`
}

/** 指标数字：保留 4 位有效小数；没有该指标时返回占位符而不是 0。 */
export function formatMetric(value?: number | null, digits = 4): string {
  if (value == null || typeof value !== 'number' || !isFinite(value)) return '-'
  return value.toFixed(digits)
}

/** 时间戳（秒或毫秒）转本地时间字符串。 */
export function formatTime(value?: number | string | null): string {
  if (value == null || value === '') return '-'
  const d = typeof value === 'number' ? new Date(value < 1e12 ? value * 1000 : value) : new Date(value)
  if (isNaN(d.getTime())) return String(value)
  return d.toLocaleString('zh-CN', { hour12: false })
}
