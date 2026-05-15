import os
import requests


class RemoteLLMBackend:
    def __init__(self, api_url=None, model_name=None, api_key=None):
        self.api_url = api_url or os.environ.get("CRITILIGHT_LLM_API_URL")
        self.model_name = model_name or os.environ.get("CRITILIGHT_LLM_API_MODEL")
        self.api_key = api_key or os.environ.get("CRITILIGHT_LLM_API_KEY")

    def infer(self, prompt):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()
