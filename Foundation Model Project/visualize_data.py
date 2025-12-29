import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from wordcloud import WordCloud

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('2DUSFM.csv', encoding='utf-8')

# 创建输出文件夹
import os
os.makedirs('visualizations', exist_ok=True)

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ============= 图1: 年份分布 =============
fig, ax = plt.subplots(figsize=(10, 6))
year_counts = df['年份'].value_counts().sort_index()
bars = ax.bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_xlabel('年份', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
ax.set_ylabel('文章数量', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
ax.set_title('超声基础模型研究文章年份分布', fontsize=16, fontweight='bold', pad=20, fontfamily='Microsoft YaHei')
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/1_年份分布.png', bbox_inches='tight', dpi=300)
plt.close()
print("✓ 图1: 年份分布已生成")

# ============= 图2: 期刊发文量统计 =============
fig, ax = plt.subplots(figsize=(14, 10))
# 统计各期刊发文量
journal_counts = df['期刊'].value_counts().sort_values(ascending=True)

# 处理期刊名称换行（长度超过30个字符时换行）
def wrap_journal_name(name, max_length=30):
    if len(name) <= max_length:
        return name
    # 在空格、逗号、冒号等位置换行
    words = name.replace(',', ', ').replace(':', ': ').split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

wrapped_names = [wrap_journal_name(name) for name in journal_counts.index]

# 使用渐变色
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(journal_counts)))
bars = ax.barh(range(len(journal_counts)), journal_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(journal_counts)))
ax.set_yticklabels(wrapped_names, fontsize=12, fontfamily='Microsoft YaHei', fontweight='bold')
ax.set_xlabel('发文数量', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
ax.set_title('各期刊发文量统计', fontsize=16, fontweight='bold', pad=20, fontfamily='Microsoft YaHei')
ax.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, count in enumerate(journal_counts.values):
    ax.text(count + max(journal_counts.values)*0.02, i, f'{count}',
            va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/2_期刊发文量.png', bbox_inches='tight', dpi=300)
plt.close()
print("✓ 图2: 期刊发文量统计已生成")

# ============= 图3: 数据量对比 =============
fig, ax = plt.subplots(figsize=(12, 8))
df_data = df[df['数据量'].notna()].copy()

# 提取数值（处理"万"单位）
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

df_data['数据量_万'] = df_data['数据量'].apply(extract_data_volume)
df_data = df_data[df_data['数据量_万'].notna()].sort_values('数据量_万', ascending=True)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_data)))
bars = ax.barh(range(len(df_data)), df_data['数据量_万'], color=colors, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(df_data)))
# Y轴标签（文章名称）- 明确指定字体
ax.set_yticklabels(df_data['文章'].str[:40] + '...', fontsize=9, fontfamily='Microsoft YaHei')
# X轴标签 - 明确指定字体
ax.set_xlabel('数据量 (万张图像)', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
# 标题 - 明确指定字体
ax.set_title('各研究使用的超声图像数据量对比', fontsize=16, fontweight='bold', pad=20, fontfamily='Microsoft YaHei')
ax.grid(axis='x', alpha=0.3)

# 添加数值标签 - 明确指定字体
for i, (idx, row) in enumerate(df_data.iterrows()):
    ax.text(row['数据量_万'] + max(df_data['数据量_万'])*0.02, i, 
            f"{row['数据量_万']:.1f}万",
            va='center', fontsize=9, fontweight='bold',
            fontfamily='Microsoft YaHei')

plt.tight_layout()
plt.savefig('visualizations/3_数据量对比.png', bbox_inches='tight', dpi=300)
plt.close()
print("✓ 图3: 数据量对比已生成")

# ============= 图4: 方法使用频率(词云-中英对照) =============
methods = df['用到的方法'].dropna()

# 统计关键方法出现次数，使用中英文对照标签
method_keywords = {
    'MAE 掩码自编码器': ['MAE', '掩码自编码器', 'Masked Autoencod'],
    'MIM 掩码图像建模': ['MIM', '掩码图像建模', 'Masked Image Modeling'],
    'Contrastive Learning 对比学习': ['对比学习', 'Contrastive'],
    'Federated Learning 联邦学习': ['联邦学习', 'Federated'],
    'Supervised Learning 有监督学习': ['有监督', 'Supervised'],
    'SAM 分割模型': ['SAM'],
    'Transformer 变换器': ['Transformer'],
    'Self-Supervised 自监督学习': ['自监督', 'Self-Supervised'],
    'CNN 卷积神经网络': ['CNN'],
    'Fine-tuning 微调': ['微调', 'Fine-tuning', 'Finetuning']
}

method_counts = {}
for method_name, keywords in method_keywords.items():
    count = 0
    for method_text in methods:
        if any(keyword in str(method_text) for keyword in keywords):
            count += 1
    if count > 0:
        method_counts[method_name] = count

# 创建词云
fig, ax = plt.subplots(figsize=(16, 10))

# 自定义蓝到红渐变色函数，低频为蓝色，高频为红色
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# 获取频率的最大最小值，用于归一化
max_freq = max(method_counts.values())
min_freq = min(method_counts.values())

# 创建颜色映射
colors_list = ['#0066CC', '#4D94FF', '#FF6B6B', '#CC0000']  # 深蓝 -> 浅蓝 -> 浅红 -> 深红
custom_cmap = LinearSegmentedColormap.from_list('blue_to_red', colors_list)

# 自定义颜色函数：根据频率返回颜色
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # 获取该词的频率
    freq = method_counts.get(word, min_freq)
    # 归一化到0-1之间
    normalized_freq = (freq - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0.5
    # 从colormap中获取对应颜色
    color = custom_cmap(normalized_freq)
    # 转换为RGB格式（0-255）
    return tuple(int(c * 255) for c in color[:3])

# 设置词云参数 - 使用支持中文的字体
wordcloud = WordCloud(
    width=1100, 
    height=600,
    background_color='white',
    color_func=color_func,  # 使用自定义颜色函数
    relative_scaling=0.6,  # 相对缩放，控制大小差异
    min_font_size=28,
    max_font_size=100,
    font_path='C:/Windows/Fonts/msyh.ttc',  # 使用微软雅黑字体
    prefer_horizontal=0.6,  # 水平方向的词的比例
    scale=3,
    collocations=False,
    margin=15,  # 词之间的间距
    max_words=50,  # 限制最大词数
    random_state=42  # 固定随机种子，确保布局一致性
).generate_from_frequencies(method_counts)

ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations/4_方法使用频率.png', bbox_inches='tight', dpi=300, pad_inches=0)
plt.close()
print("✓ 图4: 方法使用频率词云(中英对照)已生成")

# ============= 图5: 数据来源器官统计 =============
fig, ax = plt.subplots(figsize=(12, 8))
organ_data = df['数据种类'].dropna()

# 提取器官数量
organ_counts = []
organ_labels = []
for idx, row in df.iterrows():
    if pd.notna(row['数据种类']):
        data_type = str(row['数据种类'])
        article_name = str(row['文章'])[:35] + '...'
        if '器官' in data_type or '解剖结构' in data_type or '模态' in data_type:
            import re
            numbers = re.findall(r'\d+', data_type)
            if numbers:
                organ_counts.append(int(numbers[0]))
                organ_labels.append(article_name)

if organ_counts:
    sorted_indices = np.argsort(organ_counts)
    organ_counts = [organ_counts[i] for i in sorted_indices]
    organ_labels = [organ_labels[i] for i in sorted_indices]
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(organ_counts)))
    bars = ax.barh(range(len(organ_counts)), organ_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(organ_counts)))
    ax.set_yticklabels(organ_labels, fontsize=9, fontfamily='Microsoft YaHei')
    ax.set_xlabel('器官/解剖结构数量', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
    ax.set_title('各研究涵盖的器官/解剖结构数量', fontsize=16, fontweight='bold', pad=20, fontfamily='Microsoft YaHei')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, count in enumerate(organ_counts):
        ax.text(count + max(organ_counts)*0.02, i, f'{count}',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/5_器官数量统计.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图5: 器官数量统计已生成")

# ============= 图6: 综合散点图 - 数据量vs影响因子 =============
fig, ax = plt.subplots(figsize=(12, 8))
df_scatter = df.copy()
df_scatter['数据量_万'] = df_scatter['数据量'].apply(extract_data_volume)
df_scatter['影响因子_num'] = pd.to_numeric(df_scatter['影响因子'], errors='coerce')
df_scatter = df_scatter[df_scatter['数据量_万'].notna() & df_scatter['影响因子_num'].notna()]

scatter = ax.scatter(df_scatter['数据量_万'], df_scatter['影响因子_num'], 
                    s=300, c=df_scatter['年份'], cmap='coolwarm', 
                    alpha=0.7, edgecolors='black', linewidth=2)

# 添加文章标签
for idx, row in df_scatter.iterrows():
    ax.annotate(row['文章'][:20] + '...', 
                (row['数据量_万'], row['影响因子_num']),
                fontsize=8, alpha=0.8, 
                xytext=(5, 5), textcoords='offset points',
                fontfamily='Microsoft YaHei')

ax.set_xlabel('数据量 (万张图像)', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
ax.set_ylabel('期刊影响因子', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
ax.set_title('数据量与期刊影响因子关系图', fontsize=16, fontweight='bold', pad=20, fontfamily='Microsoft YaHei')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('年份', fontsize=12, fontweight='bold', fontfamily='Microsoft YaHei')

plt.tight_layout()
plt.savefig('visualizations/6_数据量vs影响因子.png', bbox_inches='tight', dpi=300)
plt.close()
print("✓ 图6: 数据量vs影响因子散点图已生成")

# ============= 图7: 文章-期刊-年份-影响因子综合可视化(紧凑型气泡图) =============
fig, ax = plt.subplots(figsize=(10, 9))

# 准备数据
df_bubble = df[df['文章'].notna()].copy()
df_bubble['影响因子_num'] = pd.to_numeric(df_bubble['影响因子'], errors='coerce')

# 处理NaN影响因子（arxiv等预印本），设置为0以便显示
df_bubble['影响因子_display'] = df_bubble['影响因子_num'].fillna(0)
df_bubble['是预印本'] = df_bubble['影响因子'].isna() | (df_bubble['影响因子'] == 'NaN')

# 按期刊和年份排序，为每个文章分配唯一的Y轴位置
df_bubble = df_bubble.sort_values(['期刊', '年份'])
df_bubble['y_pos'] = range(len(df_bubble))

# 气泡大小根据影响因子决定
df_bubble['bubble_size'] = df_bubble.apply(
    lambda row: 400 if row['是预印本'] else (row['影响因子_display'] * 50 + 400), 
    axis=1
)

# 绘制气泡图
for idx, row in df_bubble.iterrows():
    if row['是预印本']:
        # 预印本用虚线边框圆表示
        ax.scatter(row['年份'], row['y_pos'], 
                  s=row['bubble_size'], 
                  c='#E8E8E8', 
                  edgecolors='#888888',
                  linewidth=2.5,
                  alpha=0.7,
                  marker='o',
                  linestyle='--',
                  zorder=2)
    else:
        # 正式期刊用实心彩色气泡，颜色根据影响因子
        norm_if = row['影响因子_display'] / 16  # 归一化到最大IF约16
        color = plt.cm.RdYlGn(min(norm_if * 0.7 + 0.25, 0.95))
        ax.scatter(row['年份'], row['y_pos'], 
                  s=row['bubble_size'], 
                  c=[color], 
                  edgecolors='black',
                  linewidth=2,
                  alpha=0.85,
                  zorder=3)
    
    # 添加文章标题和期刊信息
    article_name = row['文章'][:28] if len(row['文章']) <= 28 else row['文章'][:25] + '...'
    
    # 影响因子标签
    if row['是预印本']:
        if_label = f"{row['期刊']} (Preprint)"
    else:
        if_label = f"{row['期刊']} (IF={row['影响因子_display']:.1f})"
    
    # 右侧添加详细信息
    ax.text(row['年份'] + 0.05, row['y_pos'], 
            f"  {article_name}",
            fontsize=9.5, 
            alpha=0.9,
            ha='left',
            va='center',
            fontweight='bold',
            fontfamily='Microsoft YaHei',
            color='#2C3E50')
    
    ax.text(row['年份'] + 0.05, row['y_pos'] - 0.3, 
            f"  {if_label}",
            fontsize=8, 
            alpha=0.75,
            ha='left',
            va='center',
            fontfamily='Microsoft YaHei',
            style='italic',
            color='#7F8C8D')

# 设置X轴
years = sorted(df_bubble['年份'].unique())
ax.set_xticks(years)
ax.set_xticklabels(years, fontsize=13, fontweight='bold')
ax.set_xlabel('发表年份', fontsize=15, fontweight='bold', fontfamily='Microsoft YaHei', labelpad=10)
ax.set_xlim(min(years) - 0.3, max(years) + 1.4)

# 隐藏Y轴刻度，更简洁
ax.set_yticks([])
ax.set_ylim(-0.8, len(df_bubble) - 0.2)

# 标题
ax.set_title('文章·期刊·年份·影响因子分布情况', 
            fontsize=17, fontweight='bold', pad=20, fontfamily='Microsoft YaHei',
            color='#2C3E50')

# 添加垂直分隔线
for year in years:
    ax.axvline(year, color='gray', alpha=0.15, linestyle='--', linewidth=1, zorder=1)

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
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
ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5, 
         frameon=True, fancybox=True, shadow=True, 
         prop={'family': 'Microsoft YaHei', 'size': 9.5},
         framealpha=0.95)

# 添加背景色
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# 调整布局
plt.tight_layout()
plt.savefig('visualizations/7_综合全景图.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ 图7: 文章-期刊-年份-影响因子综合全景图已生成")

# ============= 生成统计摘要 =============
print("\n" + "="*60)
print("数据统计摘要")
print("="*60)
print(f"总文章数: {len(df[df['文章'].notna()])}")
print(f"年份范围: {df['年份'].min()} - {df['年份'].max()}")
print(f"期刊总数: {df['期刊'].nunique()}")
print(f"平均数据量: {df_data['数据量_万'].mean():.1f}万张")
print(f"数据量范围: {df_data['数据量_万'].min():.1f}万 - {df_data['数据量_万'].max():.1f}万")
print("="*60)
print("\n✅ 所有可视化图表已生成在 'visualizations' 文件夹中!")

# ============= 图8: 数据量与器官数量综合条形图 =============
fig, ax = plt.subplots(figsize=(9, 16))

# 准备数据
df_combined = df.copy()

# 提取数据量
df_combined['数据量_万'] = df_combined['数据量'].apply(extract_data_volume)

# 提取器官数量
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

df_combined['器官数量'] = df_combined['数据种类'].apply(extract_organ_count)

# 筛选有效数据（至少有数据量或器官数量其中一个）
df_combined = df_combined[
    (df_combined['数据量_万'].notna()) | (df_combined['器官数量'].notna())
].copy()

# 填充缺失值为0以便绘图
df_combined['数据量_万'].fillna(0, inplace=True)
df_combined['器官数量'].fillna(0, inplace=True)

# 按年份和文章排序
df_combined = df_combined.sort_values(['年份', '文章'])
df_combined['x_pos'] = range(len(df_combined))

# 设置条形图的偏移量，使两个条形图错开
bar_width = 0.35
offset = 0.2

# 归一化处理：将器官数量转换为与数据量相同的比例尺度
max_data = df_combined['数据量_万'].max()
max_organ = df_combined['器官数量'].max()
# 将器官数量按比例缩放到数据量的范围
scale_factor = max_data / max_organ if max_organ > 0 else 1
df_combined['器官数量_scaled'] = df_combined['器官数量'] * scale_factor

# 绘制数据量条形图（左侧偏移）
bars1 = ax.bar(df_combined['x_pos'] - offset, df_combined['数据量_万'], 
               width=bar_width, alpha=0.85, 
               color='#3498DB', edgecolor='black', linewidth=1.5,
               label='数据量 (万张图像)')

# 绘制器官数量条形图（右侧偏移，同一侧向上）
bars2 = ax.bar(df_combined['x_pos'] + offset, df_combined['器官数量_scaled'], 
               width=bar_width, alpha=0.85,
               color='#E74C3C', edgecolor='black', linewidth=1.5,
               label='器官/解剖结构数量')

# 设置X轴标签（文章名称）- 精简显示
x_labels = []
for idx, row in df_combined.iterrows():
    # 提取文章关键词，去掉冗长部分
    article = row['文章']
    # 常见简化：去掉副标题、缩短长标题
    if ':' in article:
        article = article.split(':')[0]  # 只保留主标题
    if len(article) > 20:
        article = article[:17] + '...'
    year = int(row['年份'])
    x_labels.append(f"[{year}]\n{article}")

ax.set_xticks(df_combined['x_pos'])
ax.set_xticklabels(x_labels, fontsize=10, fontfamily='Microsoft YaHei', 
                   fontweight='bold', rotation=45, ha='right')

# 设置Y轴（显示双刻度）
ax.set_ylabel('数据量 (万张图像)', fontsize=14, fontweight='bold', 
              fontfamily='Microsoft YaHei', color='#3498DB')
ax.tick_params(axis='y', labelsize=11, labelcolor='#3498DB')

# 创建双Y轴刻度标签
# 主刻度：数据量
data_ticks = np.linspace(0, max_data, 6)
ax.set_yticks(data_ticks)
ax.set_yticklabels([f'{x:.0f}' for x in data_ticks], color='#3498DB', fontweight='bold')

# 添加第二Y轴标签（器官数量）
ax2 = ax.secondary_yaxis('right')
organ_tick_values = np.linspace(0, max_data, 6)
organ_tick_labels = [f'{int(x/scale_factor)}' for x in organ_tick_values]
ax2.set_yticks(organ_tick_values)
ax2.set_yticklabels(organ_tick_labels, color='#E74C3C', fontweight='bold')
ax2.set_ylabel('器官/解剖结构数量', fontsize=14, fontweight='bold', 
               fontfamily='Microsoft YaHei', color='#E74C3C')

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max_data * 1.15)

# 添加数值标签
for i, (idx, row) in enumerate(df_combined.iterrows()):
    # 数据量标签（对应左侧偏移的条）
    if row['数据量_万'] > 0:
        ax.text(row['x_pos'] - offset, row['数据量_万'] + max_data * 0.02, 
                f"{row['数据量_万']:.1f}",
                ha='center', fontsize=9, fontweight='bold',
                fontfamily='Microsoft YaHei', color='#2C3E50')
    
    # 器官数量标签（对应右侧偏移的条）
    if row['器官数量'] > 0:
        ax.text(row['x_pos'] + offset, row['器官数量_scaled'] + max_data * 0.02, 
                f"{int(row['器官数量'])}",
                ha='center', fontsize=9, fontweight='bold',
                fontfamily='Microsoft YaHei', color='#2C3E50')

# 标题
ax.set_title('数据量与器官覆盖范围综合对比', 
             fontsize=17, fontweight='bold', pad=20, 
             fontfamily='Microsoft YaHei', color='#2C3E50')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498DB', edgecolor='black', label='数据量 (万张图像)'),
    Patch(facecolor='#E74C3C', edgecolor='black', label='器官/解剖结构数量')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
          frameon=True, fancybox=True, shadow=True,
          prop={'family': 'Microsoft YaHei', 'size': 11},
          framealpha=0.95)

# 设置背景色
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('visualizations/8_数据量与器官数量综合对比.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ 图8: 数据量与器官数量综合条形图已生成")

print("\n✅ 所有可视化图表已生成在 'visualizations' 文件夹中!")
