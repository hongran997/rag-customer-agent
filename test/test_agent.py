import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from core.agent.hallucination_guard import HallucinationGuard
from core.agent.intent_parser import IntentParser
from core.agent.retrieval_augmentor import RetrievalAugmentor


class TestIntentParser:
    def test_parse_consult_intent(self):
        parser = IntentParser()
        with patch.object(parser, "_parse_json") as mock_parse:
            mock_parse.return_value = {
                "intent_type": "business_consult",
                "need_retrieve": True,
                "query_rewrite": "优化后的查询",
                "reason": "业务咨询",
            }
            result = parser.parse("产品退货流程")
            assert result["intent_type"] == "business_consult"
            assert result["need_retrieve"] is True

    def test_parse_chitchat(self):
        parser = IntentParser()
        with patch.object(parser, "_parse_json") as mock_parse:
            mock_parse.return_value = {
                "intent_type": "chitchat",
                "need_retrieve": False,
                "query_rewrite": "",
                "reason": "闲聊",
            }
            result = parser.parse("今天天气真好啊")
            assert result["intent_type"] == "chitchat"
            assert result["need_retrieve"] is False

    def test_parse_fallback(self):
        parser = IntentParser()
        with patch.object(parser, "parse") as mock_parse:
            mock_parse.side_effect = Exception("API Error")
            result = parser.parse("测试")
            assert result["intent_type"] == "business_consult"
            assert result["need_retrieve"] is True


class TestHallucinationGuard:
    def test_validate_source_valid(self):
        guard = HallucinationGuard()
        valid = guard._validate_source(
            answer="产品退货需要提供订单号",
            context="产品退货需要提供订单号和购买凭证",
            sources=["faq.pdf"],
        )
        assert valid is True

    def test_validate_source_invalid(self):
        guard = HallucinationGuard()
        valid = guard._validate_source(
            answer="我的手机屏幕摔碎了怎么办",
            context="产品退货需要提供订单号和购买凭证",
            sources=["faq.pdf"],
        )
        assert valid is False

    def test_fallback_response(self):
        guard = HallucinationGuard()
        result = guard._fallback_response("test_error")
        assert result["is_hallucination"] is True
        assert "暂无相关资料" in result["answer"]
        assert result["error"] == "test_error"

    def test_generate_no_context(self):
        guard = HallucinationGuard()
        result = guard.generate(
            query="测试问题",
            context="",
        )
        assert result["is_hallucination"] is True


class TestRetrievalAugmentor:
    def test_format_context_empty(self):
        augmentor = RetrievalAugmentor()
        text = augmentor.format_context([])
        assert text == ""

    def test_format_context_with_results(self):
        augmentor = RetrievalAugmentor()
        results = [
            {"text_chunk": "片段一", "source_doc": "doc1.pdf", "fusion_score": 0.9},
            {"text_chunk": "片段二", "source_doc": "doc2.pdf", "fusion_score": 0.8},
        ]
        text = augmentor.format_context(results)
        assert "片段一" in text
        assert "doc1.pdf" in text
        assert "片段二" in text
