import json
import re
from typing import List, Dict, Optional, Any
from core.agent.llm_client import llm_client
from utils.logger import logger


HALLUCINATION_SYSTEM_PROMPT = """你是一个严格的企业客服知识库问答助手。

【核心约束 - 你必须严格遵守】
1. 溯源约束：你的回答必须严格基于下面提供的"知识库参考片段"，禁止编造任何不在参考中的内容
2. 如果参考片段不足以回答用户问题，必须明确说明"暂无相关资料"
3. 输出格式约束：必须输出严格的 JSON 格式

【知识库参考片段】
{context}

【输出格式】
你必须只输出以下 JSON 结构，不要输出任何其他内容：
{{
    "answer": "基于知识库片段的回答。如果无法回答，请说'暂无相关资料'",
    "reference_source": ["引用的知识片段来源文档名"],
    "is_hallucination": false,
    "suggestion": "如果用户问题超出知识库范围，建议如何补充知识库"
}}

【注意】
- answer 字段必须用中文
- reference_source 列出实际引用的文档名数组
- is_hallucination 始终为 false（系统会自动后校验）
- suggestion 可为空字符串
"""


class HallucinationGuard:
    def __init__(self):
        self.system_prompt_template = HALLUCINATION_SYSTEM_PROMPT

    def generate(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        business_type: str = "",
    ) -> Dict[str, Any]:
        system_prompt = self.system_prompt_template.format(context=context)
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history:
                messages.append(msg)

        messages.append({"role": "user", "content": query})

        try:
            raw = llm_client.chat(messages, temperature=0.1)
            result = self._parse_and_validate(raw, context)
            logger.info(
                f"幻觉约束推理完成 | "
                f"溯源: {result.get('reference_source', [])} | "
                f"幻觉: {result.get('is_hallucination', True)}"
            )
            return result
        except Exception as e:
            logger.error(f"幻觉约束推理失败: {e}")
            return self._fallback_response(error=str(e))

    def _parse_and_validate(
        self, raw: str, context: str
    ) -> Dict[str, Any]:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return self._fallback_response(error="json_parse_error")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError:
            return self._fallback_response(error="json_decode_error")

        answer = result.get("answer", "").strip()
        sources = result.get("reference_source", [])

        has_valid_source = self._validate_source(answer, context, sources)
        if not has_valid_source:
            result["is_hallucination"] = True
            result["answer"] = "暂无相关资料，建议您联系人工客服咨询。"
            result["suggestion"] = "需要补充相关业务知识库文档"

        return result

    def _validate_source(
        self, answer: str, context: str, sources: list
    ) -> bool:
        if not answer or answer == "暂无相关资料":
            return True

        if not context:
            return False

        answer_keywords = set(re.findall(r'[\u4e00-\u9fff\w]{2,}', answer))
        context_keywords = set(re.findall(r'[\u4e00-\u9fff\w]{2,}', context))

        if not answer_keywords:
            return False

        overlap = answer_keywords & context_keywords
        ratio = len(overlap) / len(answer_keywords) if answer_keywords else 0

        return ratio >= 0.3

    def _fallback_response(self, error: str = "") -> Dict[str, Any]:
        return {
            "answer": "暂无相关资料，建议您联系人工客服咨询。",
            "reference_source": [],
            "is_hallucination": True,
            "suggestion": "系统处理异常，请检查日志",
            "error": error,
        }

    def stream_generate(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        system_prompt = self.system_prompt_template.format(context=context)
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history:
                messages.append(msg)

        messages.append({"role": "user", "content": query})

        for chunk in llm_client.chat_stream(messages, temperature=0.1):
            yield chunk


hallucination_guard = HallucinationGuard()
