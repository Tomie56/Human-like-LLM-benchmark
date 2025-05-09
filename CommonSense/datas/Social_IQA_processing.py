import os
import json
from tqdm import tqdm

# 输入文件
jsonl_path = "datas/original_datas/dev.jsonl"
label_path = "datas/original_datas/dev-labels.lst"
output_path = "datas/Social_IQA_extracted.json"

# 标签映射：1/2/3 -> A/B/C
label_map = {"1": "A", "2": "B", "3": "C"}

# 读取 JSONL 样本
with open(jsonl_path, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]

# 读取标签列表
with open(label_path, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f]

# 合并并格式化为 CommonsenseQA 格式
records = []
for example, label in tqdm(zip(examples, labels), total=len(labels), desc="构造样本"):
    question = f"{example['context']} {example['question']}".strip()
    choices = {
        "label": ["A", "B", "C"],
        "text": [example["answerA"], example["answerB"], example["answerC"]]
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

print(f"✅ 已提取并保存到 {output_path}")
