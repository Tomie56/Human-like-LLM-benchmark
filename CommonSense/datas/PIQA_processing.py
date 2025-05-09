import os
import json
from tqdm import tqdm

# 输入路径
jsonl_path = "datas/original_datas/PIQA_dev.jsonl"
label_path = "datas/original_datas/PIQA_dev-labels.lst"
output_path = "datas/PIQA_extracted.json"

# 标签映射：0 -> A, 1 -> B
label_map = {"0": "A", "1": "B"}

# 读取 JSONL 数据
with open(jsonl_path, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]

# 读取标签
with open(label_path, "r", encoding="utf-8") as f:
    label_lines = [line.strip() for line in f]

# 构造格式化输出
records = []
for example, label in tqdm(zip(examples, label_lines), total=len(label_lines), desc="构建 PIQA 样本"):
    question = example["goal"]
    choices = {
        "label": ["A", "B"],
        "text": [example["sol1"], example["sol2"]]
    }
    answer_key = label_map.get(label, "")
    records.append({
        "question": question,
        "choices": choices,
        "answerKey": answer_key
    })

# 保存为 JSON
os.makedirs("datas", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存为 {output_path}")
