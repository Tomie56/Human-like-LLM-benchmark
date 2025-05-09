import pandas as pd
import matplotlib.pyplot as plt
import os

# 准备数据
data = {
    "Model": ["ERNIE", "GLM", "Hunyuan", "LLaMA", "Qwen"],
    "CommonsenseQA": [88.0, 88.4, 90.0, 84.0, 82.4],
    "PIQA": [90.4, 90.4, 90.4, 92.8, 95.6],
    "Social IQa": [86.8, 82.8, 81.2, 80.8, 62.4],
    "Commonsense-CN": [82.4, 84.2, 85.0, 80.4, 74.4]
}

df = pd.DataFrame(data)

# 创建输出目录
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# 一张总体对比图（模型 x 多任务）
bar_width = 0.2
x = range(len(df["Model"]))

plt.figure(figsize=(10, 6))
for i, task in enumerate(df.columns[1:]):
    plt.bar(
        [pos + i * bar_width for pos in x],
        df[task],
        width=bar_width,
        label=task
    )

plt.xticks([pos + 1.5 * bar_width for pos in x], df["Model"])
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Cross-task Accuracy Comparison among LLMs, including Commonsense-CN")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acc_bar_cross_tasks_cn.png"))
plt.close()

# 为每个任务单独画图
for task in df.columns[1:]:
    plt.figure(figsize=(7, 4))
    plt.bar(df["Model"], df[task])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy for {task}")
    for i, acc in enumerate(df[task]):
        plt.text(i, acc + 1, f"{acc:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"acc_bar_{task.lower().replace(' ', '_')}.png"))
    plt.close()

print(f"All figures saved in folder: {output_dir}")
