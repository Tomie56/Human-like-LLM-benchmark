import os
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zhipuai import ZhipuAI

# 1. 配置 API Key
API_KEYS = {
    'glm-4-flash': 'key',
    'hunyuan-turbos-latest': 'key',
    'ernie-4.0-8k': 'key',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'key',
    'Qwen/Qwen2.5-72B-Instruct-128K': 'key'
}

# 2. 加载数据集并打印状态
print("Loading EmoBench datasets...")
with open('data_EA.json', 'r', encoding='utf-8') as f:
    data_ea = json.load(f)
with open('data_EU.json', 'r', encoding='utf-8') as f:
    data_eu = json.load(f)
print(f"Loaded {len(data_ea)} EA samples, {len(data_eu)} EU samples.")

# 3. 定义模型调用接口

def call_glm4(question: str, options: list) -> str:
    client = ZhipuAI(api_key=API_KEYS['glm-4-flash'])
    message = question + "\nOptions: " + "; ".join(options)
    resp = client.chat.completions.create(
        model='glm-4-flash',
        messages=[{"role": "user", "content": message}]
    )
    return resp.choices[0].message.content.strip()

def call_hunyuan(prompt: str) -> str:
    url = 'https://wcode.net/api/gpt/v1/chat/completions'
    headers = {
        'Authorization': f"Bearer {API_KEYS['hunyuan-turbos-latest']}",
        'Content-Type': 'application/json'
    }
    payload = {'model': 'hunyuan-turbos-latest', 'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]}
    r = requests.post(url, headers=headers, json=payload)
    return r.json()['choices'][0]['message']['content'].strip()

def call_ernie(prompt: str) -> str:
    return call_hunyuan(prompt)

def call_siliconflow(model: str, prompt: str) -> str:
    url = 'https://api.siliconflow.cn/v1/chat/completions'
    headers = {
        'Authorization': f"Bearer {API_KEYS[model]}",
        'Content-Type': 'application/json'
    }
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False}
    r = requests.post(url, headers=headers, json=payload)
    return r.json()['choices'][0]['message']['content'].strip()

# 4. 评测函数，单次调用

def evaluate_ea(models):
    print("\nStarting EA evaluation...")
    print(f"Total EA samples: {len(data_ea)}")
    records = []
    for model in models:
        print(f"\nEvaluating model (EA): {model}")
        correct = {'Problem': {}, 'Relationship': {}}
        total = {'Problem': {}, 'Relationship': {}}
        for i, item in enumerate(data_ea, 1):
            problem = item['Problem']
            relation = item['Relationship']
            prompt = f"Scenario: {item['Scenario']['en']}\nWhat action should {item['Subject']['en']} take?"
            options = item['Choices']['en']
            label = item['Label']
            # 单次 zero-shot 调用
            if model == 'glm-4-flash':
                ans = call_glm4(prompt, options)
            elif model in ['hunyuan-turbos-latest', 'ernie-4.0-8k']:
                ans = call_hunyuan(prompt)
            else:
                ans = call_siliconflow(model, prompt)
            idx = next((j for j, opt in enumerate(options) if ans.lower() in opt.lower()), None)
            print(f"[EA] Model={model}, Sample={i}/{len(data_ea)}, Answer='{ans}', Index={idx}")
            total['Problem'].setdefault(problem, 0)
            correct['Problem'].setdefault(problem, 0)
            total['Relationship'].setdefault(relation, 0)
            correct['Relationship'].setdefault(relation, 0)
            total['Problem'][problem] += 1
            total['Relationship'][relation] += 1
            if idx == label:
                correct['Problem'][problem] += 1
                correct['Relationship'][relation] += 1
        # 计算得分并打印
        scores = {}
        for dim in ['Problem', 'Relationship']:
            accs = [correct[dim][k] / total[dim][k] for k in total[dim]]
            scores[dim] = np.mean(accs) if accs else 0.0
        scores['EA'] = np.mean([scores['Problem'], scores['Relationship']])
        print(f"  {model} EA Scores -> Problem: {scores['Problem']:.3f}, Relationship: {scores['Relationship']:.3f}, Overall EA: {scores['EA']:.3f}")
        records.append({'Model': model, **scores})
    return pd.DataFrame(records)


def evaluate_eu(models):
    print("\nStarting EU evaluation...")
    print(f"Total EU samples: {len(data_eu)}")
    records = []
    for model in models:
        print(f"\nEvaluating model (EU): {model}")
        correct = {}
        total = {}
        for i, item in enumerate(data_eu, 1):
            cat = item['Category']
            prompt_e = f"Scenario: {item['Scenario']['en']}\nWhich emotion does {item['Subject']['en']} feel?"
            options_e = item['Emotion']['Choices']['en']
            label_e = item['Emotion']['Label']['en']
            prompt_c = f"Scenario: {item['Scenario']['en']}\nWhat is the cause?"
            options_c = item['Cause']['Choices']['en']
            label_c = item['Cause']['Label']['en']
            for task, prompt, options, label in [('Emotion', prompt_e, options_e, label_e), ('Cause', prompt_c, options_c, label_c)]:
                if model == 'glm-4-flash':
                    ans = call_glm4(prompt, options)
                elif model in ['hunyuan-turbos-latest', 'ernie-4.0-8k']:
                    ans = call_hunyuan(prompt)
                else:
                    ans = call_siliconflow(model, prompt)
                idx = next((j for j, opt in enumerate(options) if ans.lower() in opt.lower()), None)
                print(f"[EU] Model={model}, Sample={i}/{len(data_eu)}, Task={task}, Answer='{ans}', Index={idx}")
                total.setdefault((cat, task), 0)
                correct.setdefault((cat, task), 0)
                total[(cat, task)] += 1
                if idx is not None and options[idx] == label:
                    correct[(cat, task)] += 1
        # 计算得分并打印
        scores = {}
        cats = {k[0] for k in total}
        for cat in cats:
            accs = [correct[(cat, t)] / total[(cat, t)] for t in ['Emotion', 'Cause']]
            scores[cat] = np.mean(accs) if accs else 0.0
        scores['EU'] = np.mean(list(scores.values())) if scores else 0.0
        print(f"  {model} EU Score (avg over categories): {scores['EU']:.3f}")
        records.append({'Model': model, **scores})
    return pd.DataFrame(records)

if __name__ == '__main__':
    models = [
        'glm-4-flash',
        'hunyuan-turbos-latest',
        'ernie-4.0-8k',
        'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'Qwen/Qwen2.5-72B-Instruct-128K'
    ]
    print("\n=== Begin evaluation workflow ===")
    df_ea = evaluate_ea(models)
    print("\nEA evaluation completed. Results:\n", df_ea)
    df_eu = evaluate_eu(models)
    print("\nEU evaluation completed. Results:\n", df_eu)

    print("\nSaving results to CSV files...")
    df_ea.to_csv('results_ea.csv', index=False)
    df_eu.to_csv('results_eu.csv', index=False)

    print("\nGenerating and saving plots...")
    plt.figure(figsize=(8, 5))
    plt.bar(df_ea['Model'], df_ea['EA'])
    plt.title('Emotional Application (EA) Score by Model')
    plt.xticks(rotation=45); plt.ylabel('Score'); plt.tight_layout()
    plt.savefig('ea_scores.png')

    plt.figure(figsize=(8, 5))
    plt.bar(df_eu['Model'], df_eu['EU'])
    plt.title('Emotional Understanding (EU) Score by Model')
    plt.xticks(rotation=45); plt.ylabel('Score'); plt.tight_layout()
    plt.savefig('eu_scores.png')

    print("\n=== Evaluation complete. CSV and plots generated. ===")
