import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
        import re
        numbers = re.findall(r'\d+', data_type_str)
        if numbers:
            return int(numbers[0])
    return None

# ============= 创建嵌套布局图 =============
fig = plt.figure(figsize=(16, 10))

# 主图：时间线气泡图（占据整个画布）
ax_main = fig.add_subplot(111)

# 准备气泡图数据
df_bubble = df[df['文章'].notna()].copy()
df_bubble['影响因子_num'] = pd.to_numeric(df_bubble['影响因子'], errors='coerce')
df_bubble['影响因子_display'] = df_bubble['影响因子_num'].fillna(0)
df_bubble['是预印本'] = df_bubble['影响因子'].isna() | (df_bubble['影响因子'] == 'NaN')
df_bubble = df_bubble.sort_values(['期刊', '年份'])
df_bubble['y_pos'] = range(len(df_bubble))
df_bubble['bubble_size'] = df_bubble.apply(
    lambda row: 400 if row['是预印本'] else (row['影响因子_display'] * 50 + 400), 
    axis=1
)

# 绘制气泡图
for idx, row in df_bubble.iterrows():
    if row['是预印本']:
        ax_main.scatter(row['年份'], row['y_pos'], 
                  s=row['bubble_size'], 
                  c='#E8E8E8', 
                  edgecolors='#888888',
                  linewidth=2.5,
                  alpha=0.7,
                  marker='o',
                  zorder=2)
    else:
        norm_if = row['影响因子_display'] / 16
        color = plt.cm.RdYlGn(min(norm_if * 0.7 + 0.25, 0.95))
        ax_main.scatter(row['年份'], row['y_pos'], 
                  s=row['bubble_size'], 
                  c=[color], 
                  edgecolors='black',
                  linewidth=2,
                  alpha=0.85,
                  zorder=3)
    
    # 添加文章标题和期刊信息
    article_name = row['文章'][:28] if len(row['文章']) <= 28 else row['文章'][:25] + '...'
    
    if row['是预印本']:
        if_label = f"{row['期刊']} (Preprint)"
    else:
        if_label = f"{row['期刊']} (IF={row['影响因子_display']:.1f})"
    
    ax_main.text(row['年份'] + 0.05, row['y_pos'], 
            f"  {article_name}",
            fontsize=9.5, 
            alpha=0.9,
            ha='left',
            va='center',
            fontweight='bold',
            fontfamily='Microsoft YaHei',
            color='#2C3E50')
    
    ax_main.text(row['年份'] + 0.05, row['y_pos'] - 0.3, 
            f"  {if_label}",
            fontsize=8, 
            alpha=0.75,
            ha='left',
            va='center',
            fontfamily='Microsoft YaHei',
            style='italic',
            color='#7F8C8D')

# 设置主图X轴
years = sorted(df_bubble['年份'].unique())
ax_main.set_xticks(years)
ax_main.set_xticklabels(years, fontsize=13, fontweight='bold')
ax_main.set_xlabel('发表年份', fontsize=15, fontweight='bold', fontfamily='Microsoft YaHei', labelpad=10)
ax_main.set_xlim(min(years) - 0.3, max(years) + 1.4)

# 隐藏Y轴
ax_main.set_yticks([])
ax_main.set_ylim(-0.8, len(df_bubble) - 0.2)

# 标题
ax_main.set_title('超声基础模型研究全景图：文章分布与数据特征', 
            fontsize=18, fontweight='bold', pad=20, fontfamily='Microsoft YaHei',
            color='#2C3E50')

# 添加垂直分隔线
for year in years:
    ax_main.axvline(year, color='gray', alpha=0.15, linestyle='--', linewidth=1, zorder=1)

# 主图背景色
ax_main.set_facecolor('#FAFAFA')

# ============= 嵌套子图：数据量与器官数量对比 =============
# 在右上角空白区域创建嵌套子图
ax_inset = fig.add_axes([0.58, 0.60, 0.40, 0.35])  # [left, bottom, width, height]

# 准备数据
df_combined = df.copy()
df_combined['数据量_万'] = df_combined['数据量'].apply(extract_data_volume)
df_combined['器官数量'] = df_combined['数据种类'].apply(extract_organ_count)
df_combined = df_combined[
    (df_combined['数据量_万'].notna()) | (df_combined['器官数量'].notna())
].copy()
df_combined['数据量_万'].fillna(0, inplace=True)
df_combined['器官数量'].fillna(0, inplace=True)
df_combined = df_combined.sort_values(['年份', '文章'])

# 归一化处理
max_data = df_combined['数据量_万'].max()
max_organ = df_combined['器官数量'].max()
scale_factor = max_data / max_organ if max_organ > 0 else 1
df_combined['器官数量_scaled'] = df_combined['器官数量'] * scale_factor

# 设置条形图位置
x_positions = np.arange(len(df_combined))
bar_width = 0.35

# 绘制条形图
bars1 = ax_inset.bar(x_positions - bar_width/2, df_combined['数据量_万'], 
               width=bar_width, alpha=0.85, 
               color='#3498DB', edgecolor='black', linewidth=1.2,
               label='数据量')

bars2 = ax_inset.bar(x_positions + bar_width/2, df_combined['器官数量_scaled'], 
               width=bar_width, alpha=0.85,
               color='#E74C3C', edgecolor='black', linewidth=1.2,
               label='器官数量')

# 设置X轴标签（简化）
x_labels = []
for idx, row in df_combined.iterrows():
    article = row['文章']
    if ':' in article:
        article = article.split(':')[0]
    if len(article) > 15:
        article = article[:12] + '...'
    year = int(row['年份'])
    x_labels.append(f"[{year}]\n{article}")

ax_inset.set_xticks(x_positions)
ax_inset.set_xticklabels(x_labels, fontsize=7.5, fontfamily='Microsoft YaHei', 
                   rotation=45, ha='right')

# 设置左侧Y轴（数据量）
ax_inset.set_ylabel('数据量 (万张)', fontsize=10, fontweight='bold', 
              fontfamily='Microsoft YaHei', color='#3498DB')
ax_inset.tick_params(axis='y', labelsize=9, labelcolor='#3498DB')

# 数据量刻度
data_ticks = np.linspace(0, max_data, 5)
ax_inset.set_yticks(data_ticks)
ax_inset.set_yticklabels([f'{x:.0f}' for x in data_ticks], color='#3498DB', fontweight='bold', fontsize=9)

# 创建右侧Y轴（器官数量）
ax_inset2 = ax_inset.twinx()
organ_tick_values = np.linspace(0, max_data, 5)
organ_tick_labels = [f'{int(x/scale_factor)}' for x in organ_tick_values]
ax_inset2.set_yticks(organ_tick_values)
ax_inset2.set_yticklabels(organ_tick_labels, color='#E74C3C', fontweight='bold', fontsize=9)
ax_inset2.set_ylabel('器官数量', fontsize=10, fontweight='bold', 
               fontfamily='Microsoft YaHei', color='#E74C3C')
ax_inset2.set_ylim(0, max_data * 1.15)

ax_inset.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax_inset.set_ylim(0, max_data * 1.15)

# 子图标题
ax_inset.set_title('数据量与器官覆盖范围', 
             fontsize=11, fontweight='bold', pad=8, 
             fontfamily='Microsoft YaHei', color='#2C3E50')

# 子图背景
ax_inset.set_facecolor('#FFFFFF')
# 添加边框
for spine in ax_inset.spines.values():
    spine.set_edgecolor('#2C3E50')
    spine.set_linewidth(2)

# 子图图例（简化）
legend_elements_inset = [
    Patch(facecolor='#3498DB', edgecolor='black', label='数据量'),
    Patch(facecolor='#E74C3C', edgecolor='black', label='器官数量')
]
ax_inset.legend(handles=legend_elements_inset, loc='upper left', fontsize=8.5, 
          frameon=True, fancybox=True, shadow=True,
          prop={'family': 'Microsoft YaHei', 'size': 8.5},
          framealpha=0.95)

# ============= 主图图例 =============
legend_elements_main = [
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor=plt.cm.RdYlGn(0.9), markersize=14, 
           markeredgecolor='black', linewidth=2, label='高影响因子 (IF ≥ 10)'),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor=plt.cm.RdYlGn(0.55), markersize=12, 
           markeredgecolor='black', linewidth=2, label='中等影响因子 (5 ≤ IF < 10)'),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor=plt.cm.RdYlGn(0.3), markersize=10, 
           markeredgecolor='black', linewidth=2, label='低影响因子 (IF < 5)'),
    Line2D([0], [0], marker='o', color='w', 
           markerfacecolor='#E8E8E8', markersize=10, 
           markeredgecolor='#888888', linewidth=2.5, label='预印本')
]
ax_main.legend(handles=legend_elements_main, loc='lower left', fontsize=10, 
         frameon=True, fancybox=True, shadow=True, 
         prop={'family': 'Microsoft YaHei', 'size': 10},
         framealpha=0.95)

# 设置整体背景
fig.patch.set_facecolor('white')

# 保存图表
plt.savefig('visualizations/9_嵌套综合视图.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✅ 嵌套综合视图已生成: visualizations/9_嵌套综合视图.png")
