import json, csv, requests, numpy as np

data_ea = json.load(open('data_EA.json','r',encoding='utf-8'))
data_eu = json.load(open('data_EU.json','r',encoding='utf-8'))
API_KEY = 'key'

# 调用函数
def call_model(prompt, options=None):
    url = 'https://wcode.net/api/gpt/v1/chat/completions'
    headers = {'Authorization':f"Bearer {API_KEY}",'Content-Type':'application/json'}
    payload = {'model':'hunyuan-turbos-latest','messages':[]}
    if options:
        payload['messages'].append({'role':'user','content':prompt+" Options: "+"; ".join(options)})
    else:
        payload['messages'].append({'role':'user','content':prompt})
    r = requests.post(url,headers=headers,json=payload)
    return r.json()['choices'][0]['message']['content'].strip()

# EA
out_ea = 'results_hunyuan_ea.csv'
with open(out_ea,'w',newline='',encoding='utf-8') as f:
    csv.DictWriter(f, fieldnames=['Problem','Relationship','EA']).writeheader()
cp, tp, cr, tr = {},{},{},{}
for item in data_ea:
    p, r = item['Problem'], item['Relationship']
    pr = f"Scenario: {item['Scenario']['en']}\nWhat action should {item['Subject']['en']} take?"
    opts, lab = item['Choices']['en'], item['Label']
    ans = call_model(pr, opts)
    idx = next((i for i,o in enumerate(opts) if ans.lower() in o.lower()), None)
    tp[p]=tp.get(p,0)+1; tr[r]=tr.get(r,0)+1
    if idx==lab: cp[p]=cp.get(p,0)+1; cr[r]=cr.get(r,0)+1
acc_p=np.mean([cp[k]/tp[k] for k in tp]); acc_r=np.mean([cr[k]/tr[k] for k in tr]); ea=np.mean([acc_p,acc_r])
with open(out_ea,'a',newline='',encoding='utf-8') as f:
    csv.DictWriter(f, fieldnames=['Problem','Relationship','EA']).writerow({'Problem':acc_p,'Relationship':acc_r,'EA':ea})
print(f"Hunyuan EA done: {ea:.3f}")

# EU
out_eu = 'results_hunyuan_eu.csv'
with open(out_eu,'w',newline='',encoding='utf-8') as f:
    csv.DictWriter(f, fieldnames=['Category','EU']).writeheader()
cats = set(x['Category'] for x in data_eu)
for cat in cats:
    corr, tot = 0, 0
    for item in [x for x in data_eu if x['Category']==cat]:
        # Emotion
        pr = f"Scenario: {item['Scenario']['en']}\nWhich emotion does {item['Subject']['en']} feel?"
        opts, lab = item['Emotion']['Choices']['en'], item['Emotion']['Label']['en']
        ans = call_model(pr)
        idx = next((i for i,o in enumerate(opts) if ans.lower() in o.lower()), None)
        if idx is not None and opts[idx]==lab: corr+=1
        tot+=1
        # Cause
        pr = f"Scenario: {item['Scenario']['en']}\nWhat is the cause?"
        opts, lab = item['Cause']['Choices']['en'], item['Cause']['Label']['en']
        ans = call_model(pr)
        idx = next((i for i,o in enumerate(opts) if ans.lower() in o.lower()), None)
        if idx is not None and opts[idx]==lab: corr+=1
        tot+=1
    eu = corr/tot if tot>0 else 0
    with open(out_eu,'a',newline='',encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=['Category','EU']).writerow({'Category':cat,'EU':eu})
    print(f"Hunyuan EU: {cat} -> {eu:.3f}")