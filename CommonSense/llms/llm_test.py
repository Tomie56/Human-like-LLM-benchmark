import os
from dotenv import load_dotenv
from llm_apis import get_llm_client

load_dotenv()

models = ["qwen", "llama", "glm", "hunyuan", "ernie"]

test_messages = [
    {"role": "user", "content": "Where do people usually store socks?\nA. oven\nB. drawer"}
]

def test_all_models():
    for model_name in models:
        print(f"\n🧪 Testing {model_name.upper()}:")
        try:
            client = get_llm_client(model_name)
            response = client.chat_completion(test_messages, temperature=0.7)
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
            elif "result" in response:
                content = response["result"]
            else:
                content = str(response)
            print(f"✅ {model_name} response:\n{content}")
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")

if __name__ == "__main__":
    test_all_models()
