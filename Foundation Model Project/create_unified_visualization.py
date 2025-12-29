import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import re

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('2DUSFM.csv', encoding='utf-8')

# 数据量提取函数
def extract_data_volume(val):
    if pd.isna(val):
        return None
    val_str = str(val)
    if '万' in val_str:
        num = float(val_str.replace('万', '').strip())
        return num
    else:
        try:
            return float(val_str) / 10000
        except:
            return None

# 器官数量提取函数
def extract_organ_count(data_type):
    if pd.isna(data_type):
        return None
    data_type_str = str(data_type)
    if '器官' in data_type_str or '解剖结构' in data_type_str or '模态' in data_type_str:
        numbers = re.findall(r'\d+', data_type_str)
        if numbers:
            return int(numbers[0])
    return None

# ============= 准备数据 =============
df_viz = df[df['文章'].notna()].copy()

# 提取各维度数据
df_viz['影响因子_num'] = pd.to_numeric(df_viz['影响因子'], errors='coerce')
df_viz['数据量_万'] = df_viz['数据量'].apply(extract_data_volume)
df_viz['器官数量'] = df_viz['数据种类'].apply(extract_organ_count)

# 处理缺失值
df_viz['影响因子_display'] = df_viz['影响因子_num'].fillna(0)
df_viz['数据量_display'] = df_viz['数据量_万'].fillna(0)
df_viz['器官数量_display'] = df_viz['器官数量'].fillna(0)
df_viz['是预印本'] = df_viz['影响因子'].isna() | (df_viz['影响因子'] == 'NaN')

# ============= 分析数据分布，确定断裂点 =============
data_values = sorted(df_viz['数据量_display'].values)
organ_values = sorted(df_viz['器官数量_display'].values)
# X轴断裂点：453万远大于其他点（最大约200万）
# 将X轴分为两段：0-220万 和 430-480万
break_point_x_low = 220
break_point_x_high = 430

# Y轴断裂点：56个解剖结构远大于其他点（最大20个器官）
# 将Y轴分为两段：0-25 和 50-60
break_point_y_low = 25
break_point_y_high = 55

# ============= 创建断裂坐标轴（2x2布局）=============
# 计算子图的宽度和高度比例，使坐标轴比例尺一致
left_range = break_point_x_low + 25  # -25到220万的实际范围
right_range = 50  # 430-480万的实际范围（约50万）
width_ratio = [left_range, right_range]

bottom_range = break_point_y_low + 2  # -2到25的实际范围
top_range = 10  # 50-60的实际范围
height_ratio = [top_range, bottom_range]

fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(2, 2, width_ratios=width_ratio, height_ratios=height_ratio, 
                      wspace=0.05, hspace=0.05)

# 创建四个子图：左下、右下、左上、右上
ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_bottom_right = fig.add_subplot(gs[1, 1], sharey=ax_bottom_left)
ax_top_left = fig.add_subplot(gs[0, 0], sharex=ax_bottom_left)
ax_top_right = fig.add_subplot(gs[0, 1], sharex=ax_bottom_right, sharey=ax_top_left)

# 获取年份范围用于颜色映射
years_unique = sorted(df_viz['年份'].unique())
year_min = min(years_unique)
year_max = max(years_unique)

# 绘制气泡（在两个子图上）
# 放大系数：等比例放大所有圆圈
scale_factor = 2.0

for idx, row in df_viz.iterrows():
    # 气泡大小：影响因子（等比例放大）
    if row['影响因子_display'] > 0:
        # 影响因子越大，气泡越大
        bubble_size = (300 + (row['影响因子_display'] / 16) * 2200) * scale_factor
    else:
        bubble_size = 300 * scale_factor  # 预印本默认大小
    
    # 气泡颜色：年份（使用颜色映射）
    # 使用coolwarm色图，早期年份为蓝色，近期年份为红色
    year_norm = (row['年份'] - year_min) / (year_max - year_min) if year_max > year_min else 0.5
    color = plt.cm.coolwarm(year_norm)
    
    # 去掉边框，只保留气泡颜色
    edgecolor = 'none'
    linewidth = 0
    alpha = 0.85
    
    # 判断该点属于哪个区间（2x2布局）
    x_pos = row['数据量_display']
    y_pos = row['器官数量_display']
    
    if x_pos <= break_point_x_low and y_pos <= break_point_y_low:
        target_ax = ax_bottom_left
    elif x_pos > break_point_x_low and y_pos <= break_point_y_low:
        target_ax = ax_bottom_right
    elif x_pos <= break_point_x_low and y_pos > break_point_y_low:
        target_ax = ax_top_left
    else:  # x_pos > break_point_x_low and y_pos > break_point_y_low
        target_ax = ax_top_right
    
    # 绘制气泡 (X轴=数据量, Y轴=器官数量)
    scatter = target_ax.scatter(x_pos, y_pos, 
                        s=bubble_size, 
                        c=[color], 
                        edgecolors=edgecolor,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=3)
    
    # 添加文章标签（放在右下角）
    article_name = row['文章']
    if ':' in article_name:
        article_name = article_name.split(':')[0]
    if len(article_name) > 30:
        article_name = article_name[:27] + '...'
    
    # 标签位置：在气泡右下角，更靠近圆圈
    label_offset_x = 2
    label_offset_y = -1
    
    # 文章名称
    target_ax.text(x_pos + label_offset_x, y_pos + label_offset_y, 
            article_name,
            fontsize=12, 
            ha='left',
            va='top',
            fontweight='bold',
            fontfamily='Microsoft YaHei',
            color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8, linewidth=0.5))
    
    # 在气泡中心显示年份和影响因子（使用醒目的颜色）
    info_lines = []
    info_lines.append(f"{int(row['年份'])}")
    if row['影响因子_display'] > 0:
        info_lines.append(f"IF={row['影响因子_display']:.1f}")
    else:
        info_lines.append("Preprint")
    
    info_text = '\n'.join(info_lines)
    
    # 使用黑色文字，在所有颜色背景上都清晰可见
    text_color = 'black'
    
    target_ax.text(x_pos-label_offset_x, y_pos-label_offset_y, 
            info_text,
            fontsize=18,  # 从7.5增加到10
            ha='center',
            va='center',
            fontweight='bold',
            fontfamily='Microsoft YaHei',
            color=text_color,
            zorder=4)

# ============= 设置坐标轴 =============
max_data = df_viz['数据量_display'].max()
max_organ = df_viz['器官数量_display'].max()

# X轴范围
ax_bottom_left.set_xlim(-25, break_point_x_low)
ax_bottom_right.set_xlim(break_point_x_high, max_data + 20)
ax_top_left.set_xlim(-25, break_point_x_low)
ax_top_right.set_xlim(break_point_x_high, max_data + 20)

# Y轴范围
ax_bottom_left.set_ylim(-2, break_point_y_low)
ax_top_left.set_ylim(break_point_y_high, max_organ + 3)
ax_bottom_right.set_ylim(-2, break_point_y_low)
ax_top_right.set_ylim(break_point_y_high, max_organ + 3)

# X轴标签（只在底部显示）
ax_bottom_left.set_xlabel('数据量 (万张图像)', fontsize=16, fontweight='bold', 
                          fontfamily='Microsoft YaHei', labelpad=12)
ax_bottom_right.set_xlabel('数据量 (万张图像)', fontsize=16, fontweight='bold', 
                           fontfamily='Microsoft YaHei', labelpad=12)

# Y轴标签（只在左侧显示）
ax_bottom_left.set_ylabel('器官/解剖结构数量', fontsize=16, fontweight='bold', 
                          fontfamily='Microsoft YaHei', labelpad=12)
ax_top_left.set_ylabel('器官/解剖结构数量', fontsize=16, fontweight='bold', 
                       fontfamily='Microsoft YaHei', labelpad=12)

# 隐藏不需要的刻度标签
ax_top_left.tick_params(labelbottom=False)
ax_top_right.tick_params(labelbottom=False, labelleft=False)
ax_bottom_right.tick_params(labelleft=False)

# 添加网格
for ax in [ax_bottom_left, ax_bottom_right, ax_top_left, ax_top_right]:
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_facecolor('#FAFAFA')

# ============= 添加断裂线标记 =============
# 使用更柔和的断裂标记
d = 0.008  # 减小断裂标记的大小
break_color = '#999999'  # 使用灰色，更柔和
break_linewidth = 1.2  # 减小线宽

# X轴断裂线（底部行）
kwargs = dict(transform=ax_bottom_left.transAxes, color=break_color, clip_on=False, linewidth=break_linewidth)
ax_bottom_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax_bottom_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax_bottom_right.transAxes)
ax_bottom_right.plot((-d, +d), (-d, +d), **kwargs)
ax_bottom_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# X轴断裂线（顶部行）
kwargs = dict(transform=ax_top_left.transAxes, color=break_color, clip_on=False, linewidth=break_linewidth)
ax_top_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax_top_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax_top_right.transAxes)
ax_top_right.plot((-d, +d), (-d, +d), **kwargs)
ax_top_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Y轴断裂线（左列）
kwargs = dict(transform=ax_bottom_left.transAxes, color=break_color, clip_on=False, linewidth=break_linewidth)
ax_bottom_left.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax_top_left.transAxes)
ax_top_left.plot((-d, +d), (-d, +d), **kwargs)
ax_top_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)

# Y轴断裂线（右列）
kwargs = dict(transform=ax_bottom_right.transAxes, color=break_color, clip_on=False, linewidth=break_linewidth)
ax_bottom_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom_right.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax_top_right.transAxes)
ax_top_right.plot((-d, +d), (-d, +d), **kwargs)
ax_top_right.plot((1 - d, 1 + d), (-d, +d), **kwargs)

# 去除中间的spine
ax_bottom_left.spines['right'].set_visible(False)
ax_bottom_left.spines['top'].set_visible(False)
ax_bottom_right.spines['left'].set_visible(False)
ax_bottom_right.spines['top'].set_visible(False)
ax_top_left.spines['right'].set_visible(False)
ax_top_left.spines['bottom'].set_visible(False)
ax_top_right.spines['left'].set_visible(False)
ax_top_right.spines['bottom'].set_visible(False)

# ============= 标题 =============
fig.suptitle('2D超声基础模型研究综合视图', 
            fontsize=19, fontweight='bold', y=0.98, fontfamily='Microsoft YaHei',
            color='#2C3E50')

# ============= 图例 =============
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 创建统一的多维度图例（合并所有说明）
legend_elements = [
    # 影响因子图例（气泡大小）
    Line2D([0], [0], marker='o', color='w', label='━━━ 影响因子 (气泡大小) ━━━',
           markersize=0, linewidth=0),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor='#999999', markersize=20, 
           markeredgecolor='black', linewidth=2, label='高影响因子 (IF ≥ 10)'),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor='#999999', markersize=12, 
           markeredgecolor='black', linewidth=2, label='中等影响因子 (5-10)'),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor='#999999', markersize=7, 
           markeredgecolor='#666666', linewidth=2.5, label='预印本'),
    
    # 年份图例（气泡颜色）
    Line2D([0], [0], marker='o', color='w', label='\n━━━ 年份 (气泡颜色) ━━━',
           markersize=0, linewidth=0),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor=plt.cm.coolwarm(0.1), markersize=14, 
           markeredgecolor='black', linewidth=2, label='2024年 (蓝色)'),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor=plt.cm.coolwarm(1.0), markersize=14, 
           markeredgecolor='black', linewidth=2, label='2025年 (红色)'),
    
    # 坐标轴说明
    Line2D([0], [0], marker='o', color='w', label='\n━━━ 坐标轴说明 ━━━',
           markersize=0, linewidth=0),
    Line2D([0], [0], color='w', label='X轴 = 数据量 (万张图像)', markersize=0, linewidth=0),
    Line2D([0], [0], color='w', label='Y轴 = 器官/解剖结构数量', markersize=0, linewidth=0),
    Line2D([0], [0], color='w', label='注: X轴和Y轴均使用断裂坐标轴', markersize=0, linewidth=0),
]

ax_top_left.legend(handles=legend_elements, 
         loc='upper left', 
         fontsize=11, 
         frameon=True, 
         fancybox=True, 
         shadow=True, 
         prop={'family': 'Microsoft YaHei', 'size': 11},
         framealpha=0.95,
         title='图例说明',
         title_fontsize=12)

# 设置背景
fig.patch.set_facecolor('white')

# 保存图表
plt.tight_layout()
plt.savefig('visualizations/10_五维综合视图.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✅ 五维综合视图已生成: visualizations/10_五维综合视图.png")
print("\n维度说明:")
print("  ✓ X轴: 数据量 (万张图像)")
print("  ✓ Y轴: 器官/解剖结构数量")
print("  ✓ 气泡大小: 期刊影响因子")
print("  ✓ 气泡颜色: 发表年份")
print("  ✓ 标签: 文章名称")

