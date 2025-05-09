import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from llms.llm_apis import get_llm_client
import random

# 加载 .env 中的 API 密钥
load_dotenv()

# 中文版数据路径
DATA_PATH = "datas/Commonsense-CN_extracted.json"

# 加载数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    data = random.sample(data, min(250, len(data)))  # 最多取 250 条

# 构造中文 prompt
def build_zh_prompt(question, choices):
    labels = choices["label"]
    texts = choices["text"]
    prompt = f"{question}\n"
    for label, text in zip(labels, texts):
        prompt += f"{label}. {text}\n"
    prompt += "\n请选择最合适的选项，仅返回选项字母即可（例如 A、B、C...）。\n答案："
    return prompt

# 模型验证函数
def evaluate_model_zh(model_name):
    client = get_llm_client(model_name)
    predictions = []
    correct = 0

    for item in tqdm(data, desc=f"Evaluating {model_name} on CN"):
        prompt = build_zh_prompt(item["question"], item["choices"])
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            if "choices" in response:
                model_output = response["choices"][0]["message"]["content"].strip()
            elif "result" in response:
                model_output = response["result"].strip()
            else:
                model_output = str(response).strip()

            predicted = model_output[0].upper()
            label = item["answerKey"]
            is_correct = predicted == label
            if is_correct:
                correct += 1

            predictions.append({
                "question": item["question"],
                "answerKey": label,
                "model_output": model_output,
                "predicted": predicted,
                "correct": is_correct
            })

        except Exception as e:
            predictions.append({
                "question": item["question"],
                "answerKey": item.get("answerKey", ""),
                "model_output": f"[ERROR: {str(e)}]",
                "predicted": "?",
                "correct": False
            })

    # 结果保存
    os.makedirs("results/Commonsense-CN", exist_ok=True)
    with open(f"results/Commonsense-CN/{model_name}_cnqa.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    acc = correct / len([x for x in predictions if x["answerKey"]])
    acc_data = {
        "model": model_name,
        "task": "cnqa",
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(predictions)
    }
    with open(f"results/Commonsense-CN/acc_{model_name}_cnqa.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2)

    print(f"✅ {model_name} Accuracy (Commonsense-CN): {acc:.2%} ({correct}/{len(predictions)})")

# 要评估的模型名称
models = ["qwen", "llama", "glm", "hunyuan", "ernie"]

# 主程序入口
if __name__ == "__main__":
    for model in models:
        evaluate_model_zh(model)
