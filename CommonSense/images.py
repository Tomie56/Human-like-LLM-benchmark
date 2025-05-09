import pandas as pd
import matplotlib.pyplot as plt

# 构造 DataFrame
data = {
    "Model": [
        "Qwen2.5-72B-Instruct",
        "Meta LLaMA 3.1-70B-Instruct",
        "ERNIE 4.0-8K",
        "GLM-4-Plus",
        "Hunyuan-Turbo S"
    ],
    "CommonsenseQA": [82.40, 84.00, 88.00, 88.40, 90.00],
    "PIQA": [95.60, 92.80, 90.40, 90.40, 90.40],
    "Social IQa": [62.40, 80.80, 86.80, 82.80, 81.20]
}
df = pd.DataFrame(data)

# 绘制表格
fig, ax = plt.subplots(figsize=(12, 3))  # 宽度控制清晰度
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

# 样式微调
table.scale(1, 2.0)  # 行高拉升
table.auto_set_font_size(False)
table.set_fontsize(12)

# 保存为高清图像
plt.tight_layout()
plt.savefig("accuracy_table_hd.png", dpi=300)  # 或 dpi=600
plt.savefig("accuracy_table_hd.pdf")  # 可插入LaTeX
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 构造 DataFrame
data = {
    "Model": [
        "Qwen2.5-72B-Instruct",
        "Meta LLaMA 3.1-70B-Instruct",
        "ERNIE 4.0-8K",
        "GLM-4-Plus",
        "Hunyuan-Turbo S"
    ],
    "CommonsenseQA": [82.40, 84.00, 88.00, 88.40, 90.00],
    "PIQA": [95.60, 92.80, 90.40, 90.40, 90.40],
    "Social IQa": [62.40, 80.80, 86.80, 82.80, 81.20]
}
df = pd.DataFrame(data)

# 绘制表格
fig, ax = plt.subplots(figsize=(12, 3))  # 宽度控制清晰度
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

# 样式微调
table.scale(1, 2.0)  # 行高拉升
table.auto_set_font_size(False)
table.set_fontsize(12)

# 保存为高清图像
plt.tight_layout()
plt.savefig("accuracy_table_hd.png", dpi=300)  # 或 dpi=600
plt.savefig("accuracy_table_hd.pdf")  # 可插入LaTeX
plt.show()
