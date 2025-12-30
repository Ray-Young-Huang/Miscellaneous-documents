import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件（尝试多种编码）
try:
    df = pd.read_csv('其他.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('其他.csv', encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv('其他.csv', encoding='gb2312')

# 清理数据：移除完全为空的行
df = df.dropna(how='all')

# 只保留主要列（前8列）
main_columns = ['文章', '年份', '期刊', '影响因子', '数据种类', '数据量', '用到的方法']
df_display = df[main_columns].copy()

# 处理长文本，自动换行
def wrap_text(text, width=40):
    if pd.isna(text):
        return ''
    text = str(text).strip()
    if len(text) <= width:
        return text
    # 简单的换行处理
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

# 应用文本换行
for col in df_display.columns:
    if col == '文章':
        df_display[col] = df_display[col].apply(lambda x: wrap_text(x, 35))
    elif col in ['数据种类', '用到的方法']:
        df_display[col] = df_display[col].apply(lambda x: wrap_text(x, 25))
    else:
        df_display[col] = df_display[col].apply(lambda x: '' if pd.isna(x) else str(x).strip())

# 创建图形
fig, ax = plt.subplots(figsize=(20, 12))
ax.axis('tight')
ax.axis('off')

# 创建表格
table = ax.table(cellText=df_display.values,
                colLabels=df_display.columns,
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1])

# 美化表格样式
table.auto_set_font_size(False)
table.set_fontsize(9)

# 设置表头样式
for i in range(len(df_display.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=10)
    cell.set_height(0.08)

# 设置数据行样式
for i in range(1, len(df_display) + 1):
    for j in range(len(df_display.columns)):
        cell = table[(i, j)]
        
        # 交替行颜色
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#F2F2F2')
        
        # 设置边框
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        
        # 根据列调整单元格高度和宽度
        if j == 0:  # 文章列
            cell.set_width(0.28)
            cell.set_height(0.12)
        elif j == 1:  # 年份列
            cell.set_width(0.06)
            cell.set_height(0.12)
        elif j == 2:  # 期刊列
            cell.set_width(0.15)
            cell.set_height(0.12)
        elif j == 3:  # 影响因子列
            cell.set_width(0.07)
            cell.set_height(0.12)
        elif j == 4:  # 数据种类列
            cell.set_width(0.12)
            cell.set_height(0.12)
        elif j == 5:  # 数据量列
            cell.set_width(0.10)
            cell.set_height(0.12)
        elif j == 6:  # 用到的方法列
            cell.set_width(0.22)
            cell.set_height(0.12)
        
        # 文字对齐
        cell.set_text_props(va='center')

# 设置标题
plt.title('医学影像基础模型研究文献汇总', 
          fontsize=16, 
          fontweight='bold', 
          pad=20,
          color='#2F5496')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('其他FM可视化表格.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

print("✓ 表格可视化完成！已保存为: 其他FM可视化表格.png")

# 显示图片
plt.show()
