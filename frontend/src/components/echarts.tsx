/**
 * ECharts 按需引入。
 *
 * 只注册本项目实际用到的图表与组件，避免把 echarts 全量打进产物。
 * 新增图表类型（如饼图、散点图）时，记得在这里补一次 `echarts.use([...])`，
 * 否则运行时会报「series type 未注册」。
 */
import type { ComponentProps } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'

echarts.use([
  LineChart,
  BarChart,
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  TitleComponent,
  CanvasRenderer,
])

type Props = Omit<ComponentProps<typeof ReactEChartsCore>, 'echarts'>

/** 与 echarts-for-react 的默认导出用法一致，只是注入了按需注册的 echarts 实例。 */
export default function ReactECharts(props: Props) {
  return <ReactEChartsCore echarts={echarts} {...props} />
}
