import os 
import json
from tqdm import tqdm
from dotenv import load_dotenv
from llms.llm_apis import get_llm_client

# 加载 .env 中 API keys
load_dotenv()

# 加载 Social IQa 数据（最多取前 250 个）
with open("datas/Social_IQA_extracted.json", "r", encoding="utf-8") as f:
    data = json.load(f)[:250]

# 构造 prompt（确保提示模型只返回 A/B/C）
def build_prompt(question, choices):
    labels = choices["label"]
    texts = choices["text"]
    prompt = f"{question}\n"
    for label, choice in zip(labels, texts):
        prompt += f"{label}. {choice}\n"
    prompt += "\nPlease select the most appropriate answer from A, B, or C. Just return the letter only.\nAnswer:"
    return prompt

# 执行模型评估
def evaluate_model(model_name):
    client = get_llm_client(model_name)
    predictions = []
    correct = 0

    for item in tqdm(data, desc=f"Evaluating {model_name}"):
        prompt = build_prompt(item["question"], item["choices"])
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

    # 保存结果
    os.makedirs("results", exist_ok=True)
    with open(f"results/Social_IQA/{model_name}_socialiqa.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

     # 保存准确率到单独文件
    acc = correct / len([x for x in predictions if x["answerKey"]])
    acc_data = {
        "model": model_name,
        "task": "socialiqa",
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(predictions)
    }
    with open(f"results/Social_IQA/acc_{model_name}_socialiqa.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2)

    print(f"✅ {model_name} Accuracy (Social_IQA): {acc:.2%} ({correct}/{len(predictions)})")


# 要评估的模型
models = ["qwen", "llama", "glm", "hunyuan", "ernie"]

# 执行所有模型评估
if __name__ == "__main__":
    for model in models:
        evaluate_model(model)
