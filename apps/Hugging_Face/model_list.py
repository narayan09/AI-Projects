import requests
import os
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()

# Get Hugging Face API token
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set it in .env")

class HuggingFaceLLM:
    def __init__(self, model_name, api_token, temperature=0.7, max_tokens=256, timeout=30):
        self.model_name = model_name
        self.api_token = api_token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def generate(self, prompt):
        payload = {"inputs": prompt,
                   "parameters": {"temperature": self.temperature, "max_new_tokens": self.max_tokens}}
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data[0]["generated_text"] if isinstance(data, list) else str(data)

    @classmethod
    def list_models(cls, api_token, task="text-generation", limit=10):
        url = "https://huggingface.co/api/models"
        headers = {"Authorization": f"Bearer {api_token}"}
        params = {"pipeline_tag": task, "limit": limit}
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        return [m["modelId"] for m in data]


models = HuggingFaceLLM.list_models(HF_API_TOKEN, task="text-generation", limit=5)
print(models)