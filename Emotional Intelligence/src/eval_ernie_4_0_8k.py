import json, csv, requests, numpy as np

data_ea = json.load(open('data_EA.json','r',encoding='utf-8'))
data_eu = json.load(open('data_EU.json','r',encoding='utf-8'))
API_KEY = 'key'

def call_model(prompt, options=None):
    return requests.post(
        'https://wcode.net/api/gpt/v1/chat/completions',
        headers={'Authorization':f"Bearer {API_KEY}",'Content-Type':'application/json'},
        json={
            'model':'ernie-4.0-8k',
            'messages':[{'role':'user','content': prompt + (f" Options: {'; '.join(options)}" if options else '')}]
        }
    ).json()['choices'][0]['message']['content'].strip()
# EA
out_ea='results_ernie4_ea.csv'
with open(out_ea:='results_ernie4_ea.csv','w',newline='',encoding='utf-8') as f:
    csv.DictWriter(f,['Problem','Relationship','EA']).writeheader()
pc,tp,cr,tr={},{},{},{}
for it in data_ea:
    p,r=it['Problem'],it['Relationship']
    pr=f"Scenario: {it['Scenario']['en']}\nWhat action should {it['Subject']['en']} take?"
    opts,lab=it['Choices']['en'],it['Label']
    idx=next((i for i,o in enumerate(opts) if call_model(pr,opts).lower() in o.lower()),None)
    tp[p]=tp.get(p,0)+1;tr[r]=tr.get(r,0)+1
    if idx==lab:pc[p]=pc.get(p,0)+1;cr[r]=cr.get(r,0)+1
acc_p=np.mean([pc[k]/tp[k] for k in tp]);acc_r=np.mean([cr[k]/tr[k] for k in tr]);ea=np.mean([acc_p,acc_r])
with open(out_ea,'a',newline='',encoding='utf-8') as f:
    csv.DictWriter(f,['Problem','Relationship','EA']).writerow({'Problem':acc_p,'Relationship':acc_r,'EA':ea})
print(f"ERNIE EA {ea:.3f}")

# EU
oeu='results_ernie4_eu.csv'
with open(oeu,'w',newline='',encoding='utf-8') as f:
    csv.DictWriter(f,['Category','EU']).writeheader()
for cat in set(x['Category'] for x in data_eu):
    c,t=0,0
    for it in [x for x in data_eu if x['Category']==cat]:
        for pr,opts,lab in [
            (f"Scenario: {it['Scenario']['en']}\nWhich emotion does {it['Subject']['en']} feel?", it['Emotion']['Choices']['en'], it['Emotion']['Label']['en']),
            (f"Scenario: {it['Scenario']['en']}\nWhat is the cause?", it['Cause']['Choices']['en'], it['Cause']['Label']['en'])
        ]:
            idx=next((i for i,o in enumerate(opts) if call_model(pr).lower() in o.lower()),None)
            if idx is not None and opts[idx]==lab: c+=1
            t+=1
    eu=c/t if t>0 else 0
    with open(oeu,'a',newline='',encoding='utf-8') as f:
        csv.DictWriter(f,['Category','EU']).writerow({'Category':cat,'EU':eu})
    print(f"ERNIE EU {cat}:{eu:.3f}")