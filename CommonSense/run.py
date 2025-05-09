import os
import json
import subprocess
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.dirname(__file__))
env = os.environ.copy()
env["PYTHONPATH"] = project_root

# ========== 步骤 1：运行所有 eval 脚本 ==========
eval_scripts = [
    "eval/eval_CommonsenseQA.py",
    "eval/eval_Social_IQA.py",
    "eval/eval_PIQA.py"
]

for script in eval_scripts:
    print(f"🚀 Running: {script}")
    subprocess.run(["python", script], check=True, env=env)

# ========== 步骤 2：收集所有 acc 结果 ==========
acc_results = []
results_root = "results"

for task_dir in os.listdir(results_root):
    task_path = os.path.join(results_root, task_dir)
    if os.path.isdir(task_path):
        for filename in os.listdir(task_path):
            if filename.startswith("acc_") and filename.endswith(".json"):
                with open(os.path.join(task_path, filename), "r", encoding="utf-8") as f:
                    result = json.load(f)
                    result["task"] = task_dir 
                    acc_results.append(result)

# ========== 步骤 3：每个任务单独画图 ==========
acc_by_task = {}
for item in acc_results:
    acc_by_task.setdefault(item["task"], []).append(item)

for task, entries in acc_by_task.items():
    models = [e["model"] for e in entries]
    accs = [e["accuracy"] for e in entries]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accs)
    plt.title(f"Accuracy for {task.upper()}")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc, f"{acc:.2%}", ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join("results", f"acc_bar_{task.lower()}.png")
    plt.savefig(plot_path)
    print(f"📊 Saved task-wise plot: {plot_path}")

# ========== 步骤 4：跨任务 LLM 对比图 ==========
# 聚合为：每个模型在不同任务下的准确率
model_task_map = {}
for item in acc_results:
    model = item["model"]
    task = item["task"]
    acc = item["accuracy"]
    model_task_map.setdefault(model, {})[task] = acc

tasks = sorted({item["task"] for item in acc_results})
models = sorted(model_task_map.keys())

plt.figure(figsize=(10, 6))
bar_width = 0.15
x = range(len(models))

for i, task in enumerate(tasks):
    accs = [model_task_map[model].get(task, 0) for model in models]
    plt.bar([pos + i * bar_width for pos in x], accs, width=bar_width, label=task)

plt.xticks([pos + bar_width for pos in x], models)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Cross-Task Accuracy Comparison")
plt.legend(title="Task")
plt.tight_layout()
plt.savefig("results/acc_bar_cross_tasks.png")
print("📊 Saved cross-task plot: results/acc_bar_cross_tasks.png")

# ========== 步骤 5：输出 Markdown 表格 ==========
markdown_path = "results/accuracy_summary.md"
with open(markdown_path, "w", encoding="utf-8") as f:
    f.write("| Task | Model | Accuracy |\n|------|--------|----------|\n")
    for item in sorted(acc_results, key=lambda x: (x["task"], -x["accuracy"])):
        f.write(f"| {item['task']} | {item['model']} | {item['accuracy']:.2%} |\n")

print(f"✅ Markdown summary saved: {markdown_path}")
