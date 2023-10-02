import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
x = np.linspace(0, 10, 100)
x2 = x[:80]
y_large_range = np.sin(x2) * 10
x1 = x[80:]
y_small_fluctuations = np.sin(x1) * 0.1

# 创建一个新的Figure和Axes对象
fig, ax1 = plt.subplots()

# 绘制主要的曲线在左侧Y轴
ax1.plot(x2, y_large_range, color='tab:blue', label='Main Curve')
ax1.set_xlabel('X')
ax1.set_ylabel('Main Curve', color='tab:blue')

# 创建第二个Y轴
ax2 = ax1.twinx()

# 绘制细小波动在右侧Y轴
ax2.plot(x1, y_small_fluctuations, color='tab:red', label='Small Fluctuations')
ax2.set_ylabel('Small Fluctuations', color='tab:red')

# 合并两个图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# 显示图形
plt.show()




# # 创建示例数据
# x = np.linspace(0, 10, 100)
# y_large_range = np.sin(x) * 10
# y_small_fluctuations = np.sin(x) * 0.1

# # 创建一个新的Figure和Axes对象
# fig, ax = plt.subplots()

# # 绘制主要的曲线
# ax.plot(x, y_large_range, label='Main Curve')

# # 设置次要刻度和次要网格线
# ax.yaxis.set_minor_locator(plt.MultipleLocator(1))  # 设置次要刻度间隔为1
# ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)  # 显示次要网格线

# # 创建放大的子图
# axins = ax.inset_axes([0.5, 0.2, 0.4, 0.3])  # 调整子图的位置和大小
# axins.plot(x, y_small_fluctuations, label='Small Fluctuations')

# # 显示图例
# ax.legend()

# # 显示图形
# plt.show()


fig.savefig('1.png')