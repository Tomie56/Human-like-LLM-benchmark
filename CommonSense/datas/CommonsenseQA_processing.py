from datasets import load_dataset
import json
import os
from tqdm import tqdm

# 仅加载 train 数据集
dataset = load_dataset("tau/commonsense_qa", split="train")

# 提取指定字段
records = []
for example in tqdm(dataset, desc="提取 train 样本"):
    records.append({
        "question": example["question"],
        "choices": example["choices"],
        "answerKey": example["answerKey"]
    })

# 保存为 JSON 文件
os.makedirs("datas", exist_ok=True)
with open("datas/CommonsenseQA_extracted.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("✅ 已提取并保存为 datas/CommonsenseQA_extracted.json")
