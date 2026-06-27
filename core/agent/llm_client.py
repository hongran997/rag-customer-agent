from typing import Optional, List, Dict, Any
from openai import OpenAI
from utils.logger import logger
from utils.constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMClient:
    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return resp.choices[0].message.content

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


llm_client = LLMClient()
