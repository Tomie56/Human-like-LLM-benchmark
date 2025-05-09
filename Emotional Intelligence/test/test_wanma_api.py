import requests

url = "https://wcode.net/api/account/billing/grants"

payload = {}
headers = {
  'Authorization': 'key'  
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)