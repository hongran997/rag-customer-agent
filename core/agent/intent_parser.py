import json
import re
from typing import Dict, Optional, List
from core.agent.llm_client import llm_client
from utils.logger import logger


INTENT_SYSTEM_PROMPT = """你是一个智能客服意图分析助手。请分析用户的问题，输出严格的 JSON 格式。

任务：
1. 判断用户问题类型
2. 如果需要检索，改写为更清晰的检索查询

问题类型：
- "business_consult": 业务咨询类（需要检索知识库）
- "chitchat": 闲聊无关类（直接拒绝回答）
- "follow_up": 多轮追问类（携带历史上下文，需要检索知识库）
- "unclear": 意图不明确

输出 JSON 格式（严格遵循，不要输出任何其他内容）：
{
    "intent_type": "business_consult | chitchat | follow_up | unclear",
    "need_retrieve": true | false,
    "query_rewrite": "优化后的检索查询语句",
    "reason": "意图判断的简短理由"
}

注意：
- 对于业务咨询和追问，need_retrieve 为 true，并对 query 进行同义扩写优化
- 对于闲聊，need_retrieve 为 false，query_rewrite 填空字符串
- 如果用户用词模糊，应进行合理补充扩写
"""


ALLOWED_INTENTS = {"business_consult", "chitchat", "follow_up", "unclear"}


class IntentParser:
    def parse(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]

        if history:
            history_content = self._format_history(history)
            messages.append({
                "role": "user",
                "content": f"以下为历史对话：\n{history_content}\n\n当前用户问题：{query}",
            })
        else:
            messages.append({"role": "user", "content": query})

        try:
            raw_result = llm_client.chat(messages, temperature=0.1)
            result = self._parse_json(raw_result)
            self._validate(result)
            logger.info(
                f"意图解析: {result['intent_type']} | "
                f"需检索: {result['need_retrieve']} | "
                f"改写: {result['query_rewrite']}"
            )
            return result
        except Exception as e:
            logger.warning(f"意图解析失败, 使用默认值: {e}")
            return {
                "intent_type": "business_consult",
                "need_retrieve": True,
                "query_rewrite": query,
                "reason": "parse_fallback",
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        lines = []
        for msg in history[-5:]:
            role = "用户" if msg.get("role") == "user" else "客服"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> dict:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"无法从响应中解析 JSON: {text[:200]}")

    def _validate(self, result: dict):
        if "intent_type" not in result:
            result["intent_type"] = "business_consult"
        if result["intent_type"] not in ALLOWED_INTENTS:
            result["intent_type"] = "business_consult"
        if "need_retrieve" not in result:
            result["need_retrieve"] = True
        if "query_rewrite" not in result or not result["query_rewrite"]:
            result["query_rewrite"] = ""
        return result


intent_parser = IntentParser()
