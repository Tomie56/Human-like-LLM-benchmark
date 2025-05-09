import json
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 DeepSeek-R1 模型
model_name = "deepseek-ai/deepseek-moe-rlhf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# 数据路径
input_files = [
    "datas/CommonsenseQA_extracted.json",
    "datas/PIQA_extracted.json",
    "datas/Social_IQA_extracted.json"
]

output_data = []
sample_size_per_file = 100  # 可修改为每个数据集中采样条数

def translate(text):
    prompt = f"请将以下英文内容翻译为流畅自然的中文：\n\n{text.strip()}\n\n翻译："
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output_ids = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.7)
    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated.split("翻译：")[-1].strip()

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 随机抽样
    samples = random.sample(dataset, min(len(dataset), sample_size_per_file))
    translated_samples = []

    for item in samples:
        new_item = item.copy()
        try:
            new_item['question'] = translate(item['question'])

            if 'choices' in item and 'text' in item['choices']:
                new_item['choices']['text'] = [translate(c) for c in item['choices']['text']]
        except Exception as e:
            print(f"翻译失败，跳过该项：{e}")
            continue

        translated_samples.append(new_item)
    
    return translated_samples

# 处理所有文件
for path in input_files:
    output_data.extend(process_file(path))

# 保存为新的 JSON 文件
output_path = Path("datas/Commonsense-CN_extracted.json")
with output_path.open('w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"翻译完成，共生成 {len(output_data)} 条数据，已保存至 {output_path}")
