from typing import Dict, Any, Optional, List, Generator
from core.agent.intent_parser import intent_parser
from core.agent.retrieval_augmentor import retrieval_augmentor
from core.agent.hallucination_guard import hallucination_guard
from core.agent.llm_client import llm_client
from api.session_store import session_manager
from utils.logger import logger


class RAGAgent:
    def __init__(self):
        self.intent_parser = intent_parser
        self.retrieval_augmentor = retrieval_augmentor
        self.hallucination_guard = hallucination_guard
        self.session_manager = session_manager

    def run(
        self,
        query: str,
        session_id: str,
        business_type: str = "default",
    ) -> Dict[str, Any]:
        logger.info(f"=== Agent 推理开始 [session={session_id}] ===")

        history = self.session_manager.get_relevant_context(
            session_id, query
        )

        intent = self.intent_parser.parse(query, history)
        logger.info(f"意图解析: {intent}")

        if not intent["need_retrieve"]:
            result = {
                "answer": "您好，我是客服知识库助手，仅解答业务相关问题。请咨询具体业务问题。",
                "intent_type": intent["intent_type"],
                "reference_source": [],
                "is_hallucination": False,
                "suggestion": "",
            }
            self._save_conversation(session_id, query, result)
            return result

        search_query = intent.get("query_rewrite") or query
        retrieved = self.retrieval_augmentor.retrieve_context(
            query=search_query,
            business_type=business_type,
        )

        if not retrieved:
            result = {
                "answer": "暂无相关资料，建议您联系人工客服咨询。",
                "intent_type": intent["intent_type"],
                "reference_source": [],
                "is_hallucination": False,
                "suggestion": "知识库中未找到相关内容，请补充知识库",
            }
            self._save_conversation(session_id, query, result)
            return result

        context_text = self.retrieval_augmentor.format_context(retrieved)
        result = self.hallucination_guard.generate(
            query=query,
            context=context_text,
            history=history,
            business_type=business_type,
        )

        reference_sources = list(set(
            r.get("source_doc", "") for r in retrieved if r.get("source_doc")
        ))
        result["reference_source"] = reference_sources
        result["intent_type"] = intent["intent_type"]

        self._save_conversation(session_id, query, result)

        source_summary = (
            f"\n\n---\n📎 参考来源: {', '.join(reference_sources)}"
            if reference_sources else ""
        )
        result["answer_with_source"] = result["answer"] + source_summary

        logger.info(f"=== Agent 推理完成 ===")
        return result

    def stream_run(
        self,
        query: str,
        session_id: str,
        business_type: str = "default",
    ) -> Generator[str, None, None]:
        logger.info(f"=== Agent 流式推理开始 [session={session_id}] ===")

        history = self.session_manager.get_relevant_context(
            session_id, query
        )
        intent = self.intent_parser.parse(query, history)

        if not intent["need_retrieve"]:
            yield "您好，我是客服知识库助手，仅解答业务相关问题。请咨询具体业务问题。"
            return

        search_query = intent.get("query_rewrite") or query
        retrieved = self.retrieval_augmentor.retrieve_context(
            query=search_query,
            business_type=business_type,
        )

        if not retrieved:
            yield "暂无相关资料，建议您联系人工客服咨询。"
            return

        context_text = self.retrieval_augmentor.format_context(retrieved)
        full_answer = ""

        for chunk in self.hallucination_guard.stream_generate(
            query=query, context=context_text, history=history
        ):
            full_answer += chunk
            yield chunk

        reference_sources = list(set(
            r.get("source_doc", "") for r in retrieved if r.get("source_doc")
        ))
        source_text = (
            f"\n\n---\n📎 参考来源: {', '.join(reference_sources)}"
            if reference_sources else ""
        )
        if source_text:
            yield source_text

        result = {
            "answer": full_answer,
            "intent_type": intent["intent_type"],
            "reference_source": reference_sources,
        }
        self._save_conversation(session_id, query, result)

    def _save_conversation(
        self,
        session_id: str,
        query: str,
        result: Dict[str, Any],
    ):
        self.session_manager.add_message(session_id, "user", query)
        self.session_manager.add_message(
            session_id, "assistant", result.get("answer", "")
        )


rag_agent = RAGAgent()
