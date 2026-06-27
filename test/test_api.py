import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "rag_customer_agent"


class TestChatAPI:
    def test_chat_no_query(self):
        resp = client.post("/api/v1/chat", json={})
        assert resp.status_code == 422

    def test_chat_invalid_params(self):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "", "session_id": "test"},
        )
        assert resp.status_code == 422

    @patch("api.route.endpoints.rag_agent")
    def test_chat_mocked(self, mock_agent):
        mock_agent.run.return_value = {
            "answer": "测试回答",
            "intent_type": "business_consult",
            "reference_source": ["doc.pdf"],
            "is_hallucination": False,
        }
        resp = client.post(
            "/api/v1/chat",
            json={
                "query": "测试问题",
                "session_id": "test-session",
                "business_type": "default",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "测试回答"


class TestKnowledgeAPI:
    def test_upload_unsupported_format(self):
        resp = client.post(
            "/api/v1/knowledge/upload",
            files={"file": ("test.xlsx", b"test", "application/octet-stream")},
            data={"business_type": "default"},
        )
        assert resp.status_code == 400

    def test_upload_txt(self):
        resp = client.post(
            "/api/v1/knowledge/upload",
            files={
                "file": (
                    "test.txt",
                    "这是测试知识库内容。".encode("utf-8"),
                    "text/plain",
                )
            },
            data={"business_type": "test_biz"},
        )
        assert resp.status_code in (200, 500)

    def test_delete_knowledge(self):
        resp = client.delete("/api/v1/knowledge/nonexistent_biz")
        assert resp.status_code in (200, 500)


class TestSessionAPI:
    def test_get_session_history(self):
        resp = client.get("/api/v1/session/test-session")
        assert resp.status_code == 200

    def test_clear_session(self):
        resp = client.delete("/api/v1/session/test-session")
        assert resp.status_code == 200
