import os
import json
import csv
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zhipuai import ZhipuAI

# API Key for glm-4-plus
API_KEY = 'key'
# Load EmoBench datasets
data_ea = json.load(open('data_EA.json', 'r', encoding='utf-8'))
data_eu = json.load(open('data_EU.json', 'r', encoding='utf-8'))

# call function
def call_model(prompt, options=None):
    client = ZhipuAI(api_key=API_KEY)
    if options:
        message = prompt + "\nOptions: " + "; ".join(options)
    else:
        message = prompt
    resp = client.chat.completions.create(model='glm-4-flash', messages=[{"role":"user","content":message}])
    return resp.choices[0].message.content.strip()

# Evaluate EA
def evaluate_ea():
    out_file = 'results_glm4plus_ea.csv'
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Problem','Relationship','EA'])
        writer.writeheader()
    correct_p, total_p = {}, {}
    correct_r, total_r = {}, {}
    for item in data_ea:
        prob, rel = item['Problem'], item['Relationship']
        prompt = f"Scenario: {item['Scenario']['en']}\nWhat action should {item['Subject']['en']} take?"
        opts, label = item['Choices']['en'], item['Label']
        ans = call_model(prompt, opts)
        idx = next((i for i,o in enumerate(opts) if ans.lower() in o.lower()), None)
        total_p[prob] = total_p.get(prob,0)+1
        total_r[rel] = total_r.get(rel,0)+1
        if idx==label:
            correct_p[prob] = correct_p.get(prob,0)+1
            correct_r[rel] = correct_r.get(rel,0)+1
    acc_p = np.mean([correct_p[k]/total_p[k] for k in total_p])
    acc_r = np.mean([correct_r[k]/total_r[k] for k in total_r])
    ea = np.mean([acc_p,acc_r])
    with open(out_file, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=['Problem','Relationship','EA']).writerow({
            'Problem':acc_p,'Relationship':acc_r,'EA':ea
        })

# Evaluate EU
def evaluate_eu():
    out_file = 'results_glm4plus_eu.csv'
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Category','EU'])
        writer.writeheader()
    results = []
    cats = set(item['Category'] for item in data_eu)
    for cat in cats:
        corr, tot = 0, 0
        for item in [x for x in data_eu if x['Category']==cat]:
            # Emotion
            prompt = f"Scenario: {item['Scenario']['en']}\nWhich emotion does {item['Subject']['en']} feel?"
            opts, label = item['Emotion']['Choices']['en'], item['Emotion']['Label']['en']
            ans = call_model(prompt)
            idx = next((i for i,o in enumerate(opts) if ans.lower() in o.lower()), None)
            if idx is not None and opts[idx]==label: corr+=1
            tot+=1
            # Cause
            prompt = f"Scenario: {item['Scenario']['en']}\nWhat is the cause?"
            opts, label = item['Cause']['Choices']['en'], item['Cause']['Label']['en']
            ans = call_model(prompt)
            idx = next((i for i,o in enumerate(opts) if ans.lower() in o.lower()), None)
            if idx is not None and opts[idx]==label: corr+=1
            tot+=1
        eu = corr/tot if tot>0 else 0
        with open(out_file, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=['Category','EU']).writerow({'Category':cat,'EU':eu})

if __name__=='__main__':
    evaluate_ea()
    evaluate_eu()