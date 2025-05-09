import os
import json
import requests

# 确保已设置 ERNIE_AK 和 ERNIE_SK
os.environ["ERNIE_AK"] = "xs4Gp3PEdBqoxmvsAum57z4I"
os.environ["ERNIE_SK"] = "KiaxmWWbTHHhsK3VVj3HtIZkpN9WNKQo"

# bce-v3/ALTAK-U4Dtt73WHdPWcSvG6cAeJ/4ab22ee18d57a27e7277722995bb0e2b2e0781e2

# ERNIE_AK=xs4Gp3PEdBqoxmvsAum57z4I
# ERNIE_SK=KiaxmWWbTHHhsK3VVj3HtIZkpN9WNKQo

class ErnieClient:
    def __init__(self):
        self.api_key = os.environ.get("ERNIE_AK")
        self.secret_key = os.environ.get("ERNIE_SK")
        self.token = self.get_access_token()
        self.api_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={self.token}"

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        response = requests.post(url, params=params)
        if response.status_code != 200:
            print("❌ 获取 access_token 失败:", response.text)
            return None
        return response.json().get("access_token")

    def chat_completion(self, messages, **kwargs):
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.95),
            "top_p": kwargs.get("top_p", 0.8),
            "penalty_score": kwargs.get("penalty_score", 1),
            "enable_system_memory": kwargs.get("enable_system_memory", False),
            "disable_search": kwargs.get("disable_search", False),
            "enable_citation": kwargs.get("enable_citation", False)
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return response

# ========== 测试 ==========
if __name__ == "__main__":
    client = ErnieClient()
    if not client.token:
        print("❌ 无法获取 access_token，检查 AK/SK 设置")
    else:
        print(f"✅ access_token 获取成功：{client.token[:20]}...")

        # 发起测试对话请求
        response = client.chat_completion([
            {"role": "user", "content": "1+1等于多少？"}
        ], temperature=0.5)

        try:
            data = response.json()
            print("✅ 返回结果：", json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            print("❌ 返回解析失败：", e)
            print("原始响应：", response.text)
