# 文件：five_llms_openai_style_client.py

from dotenv import load_dotenv
import os
import json
import requests

# ====== Qwen 和 LLaMA3 接口（SiliconFlow） ======
class SiliconFlowClient:
    def __init__(self, model):
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        self.model = model
        self.api_url = "https://api.siliconflow.com/v1/chat/completions"

    def chat_completion(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()


# ====== GLM 接口（智谱） ======
class GLMClient:
    def __init__(self):
        self.api_key = os.environ.get("GLM_API_KEY")
        self.model = "glm-4-plus"
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def chat_completion(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()


# ====== HunYuan 接口（腾讯） ======
class HunYuanClient:
    def __init__(self):
        self.api_key = os.environ.get("HUNYUAN_API_KEY")
        self.model = "hunyuan-turbos-latest"
        self.api_url = "https://api.hunyuan.cloud.tencent.com/v1/chat/completions"

    def chat_completion(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()


# ====== ERNIE 接口（百度千帆） ======
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
        return response.json().get("access_token")

    def chat_completion(self, messages, **kwargs):
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.95),
            "top_p": kwargs.get("top_p", 0.8),
            "penalty_score": kwargs.get("penalty_score", 1),
            "enable_system_memory": kwargs.get("enable_system_memory", False),
            "disable_search": kwargs.get("disable_search", False),
            "enable_citation": kwargs.get("enable_citation", False),
            "max_output_tokens": 256
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return response.json()


# ====== 工厂函数 ======
def get_llm_client(name):
    name = name.lower()
    if name == "qwen":
        return SiliconFlowClient("Qwen/Qwen2.5-72B-Instruct-128K")
    elif name == "llama":
        return SiliconFlowClient("meta-llama/Meta-Llama-3.1-70B-Instruct")
    elif name == "glm":
        return GLMClient()
    elif name == "hunyuan":
        return HunYuanClient()
    elif name == "ernie":
        return ErnieClient()
    else:
        raise ValueError(f"Unknown model name: {name}")
    

# ====== 示例统一调用入口 ======
if __name__ == "__main__":
    # os.environ["SILICONFLOW_API_KEY"] = "sk-ywmtflngcjcacbwgjfdfrhuimqtaseevuibfheatvfmcqvmn"
    # os.environ["GLM_API_KEY"] = "b4674b412e854a49890b3c9353ba830d.cu27XvP2swiOX1FC"
    # os.environ["HUNYUAN_API_KEY"] = "sk-aMClrpbFbcwErFdhrURT31MeeLsFyKYNa5QV3eEQUaEm93Pv"
    # os.environ["ERNIE_AK"] = "bce-v3/ALTAK-U4Dtt73WHdPWcSvG6cAeJ/4ab22ee18d57a27e7277722995bb0e2b2e0781e2"
    # os.environ["ERNIE_SK"] = "aa1dd955bac84363a8c45bee4e0aad22"
    load_dotenv()

    models = ["qwen", "llama", "glm", "hunyuan", "ernie"]
    messages = [{"role": "user", "content": "Where do you usually store your socks?\nA. oven\nB. drawer"}]

    for model_name in models:
        print(f"\n🧠 {model_name.upper()} Result:")
        client = get_llm_client(model_name)
        try:
            result = client.chat_completion(messages)
            print(result.get("choices", [{"message": {"content": result.get('result', '[No output]')}}])[0]["message"]["content"])
        except Exception as e:
            print(f"❌ Error calling {model_name}: {e}")
