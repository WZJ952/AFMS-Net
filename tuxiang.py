import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备 (根据你的表格提取)
models = [
    "SSD", "Faster-RCNN", "YOLOv5s", "YOLOv5n", "YOLOv8s",
    "YOLOv8n", "YOLOv10s", "YOLOv10n", "YOLOv11n",
    "RT-DETR-r18", "MobileNetV4", "RT-DETR-StarNet", "SDD-RT-DETR"
]

# 黄线数据 (mAP50%)
map50 = [
    73.2, 74.8, 93.0, 89.6, 93.6,
    89.8, 92.6, 89.9, 91.1,
    93.5, 91.1, 91.7, 95.8
]

# 红线数据 (FPS)
fps = [
    72.4, 58.6, 92.5, 93.6, 92.1,
    95.3, 91.8, 92.7, 94.6,
    88.1, 91.6, 90.4, 91.3
]

# 2. 生成随机误差范围 (模拟上下浮动)
# 设置随机种子保证每次画图一致
np.random.seed(42)
# 生成较小的随机浮动值 (例如 0.5 到 1.5 之间)
yerr_map = np.random.uniform(0.5, 1.5, size=len(models))
yerr_fps = np.random.uniform(0.5, 1.5, size=len(models))

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(13, 8)) # 设置长宽比，类似参考图

# 设置背景网格 (仅横向)
ax.grid(axis='y', linestyle='-', alpha=0.5, color='gray')
ax.set_axisbelow(True)

# 绘制 mAP50 (黄色，星星图标，虚线)
# color='#FFD700' (Gold) 接近参考图黄色
ax.errorbar(models, map50, yerr=yerr_map,
            fmt='*',          # 数据点标记为星星
            linestyle='--',   # 虚线
            color='#E6B800',  # 深黄色，比纯黄在白底上更清晰
            ecolor='#E6B800', # 误差棒颜色
            elinewidth=1.5,   # 误差棒线宽
            capsize=4,        # 误差棒两端横线长度
            markersize=10,    # 标记大小
            linewidth=2,      # 连线宽度
            label='mAP50 (%)')

# 绘制 FPS (红色，圆点图标，虚线)
ax.errorbar(models, fps, yerr=yerr_fps,
            fmt='o',          # 数据点标记为圆点
            linestyle='--',   # 虚线
            color='#FF4D4D',  # 红色
            ecolor='#FF4D4D', # 误差棒颜色
            elinewidth=1.5,
            capsize=4,
            markersize=8,
            linewidth=2,
            label='FPS')

# 4. 坐标轴与样式调整
# Y轴范围：由于FPS最高有91.2，建议设为95，如果强制85请改为 (40, 85)
ax.set_ylim(50, 100)

# X轴标签：为了防止重叠，稍微调整字体大小
plt.xticks(rotation=45, ha='right',fontsize=13)
plt.yticks(fontsize=11, fontweight='bold')

# 去掉顶部和右侧的边框线 (让图表看起来更干净，类似论文风格)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) # 左侧线也可以去掉，只留刻度
ax.spines['bottom'].set_linewidth(1.2)

# 图例 (放在底部居中)
# frameon=False 去掉图例边框
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
           ncol=2, frameon=False, fontsize=17)

# 布局紧凑
plt.tight_layout()

# 保存或显示
plt.show()