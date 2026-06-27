from typing import List, Dict, Optional, Any
import json
import redis
from utils.logger import logger
from utils.constants import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, SESSION_TTL


class SessionContextManager:
    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD,
        session_ttl: int = SESSION_TTL,
        max_context_tokens: int = 4096,
    ):
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=True,
        )
        self.session_ttl = session_ttl
        self.max_context_tokens = max_context_tokens
        self._test_connection()

    def _test_connection(self):
        try:
            self.redis_client.ping()
            logger.info("Redis 连接成功")
        except redis.ConnectionError:
            logger.warning("Redis 连接失败, 将使用内存模式")
            self.redis_client = None
            self._memory_store: Dict[str, list] = {}

    def _get_store(self, session_id: str) -> list:
        if self.redis_client:
            data = self.redis_client.get(f"session:{session_id}")
            return json.loads(data) if data else []
        else:
            return self._memory_store.get(session_id, [])

    def _set_store(self, session_id: str, messages: list):
        if self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(messages, ensure_ascii=False),
            )
        else:
            self._memory_store[session_id] = messages

    def get_history(
        self, session_id: str, max_turns: int = 10
    ) -> List[Dict[str, str]]:
        messages = self._get_store(session_id)
        recent = messages[-max_turns * 2:] if messages else []
        return recent

    def add_message(
        self, session_id: str, role: str, content: str
    ):
        messages = self._get_store(session_id)
        messages.append({"role": role, "content": content})
        self._set_store(session_id, messages)

    def get_relevant_context(
        self,
        session_id: str,
        current_query: str,
        max_turns: int = 5,
    ) -> List[Dict[str, str]]:
        messages = self.get_history(session_id, max_turns)
        if not messages:
            return []

        total_tokens = sum(len(m.get("content", "")) for m in messages)
        if total_tokens > self.max_context_tokens:
            messages = messages[-4:]
            logger.info(
                f"上下文超长, 裁剪至最近 {len(messages)} 条"
            )

        return messages

    def clear_session(self, session_id: str):
        if self.redis_client:
            self.redis_client.delete(f"session:{session_id}")
        elif session_id in self._memory_store:
            del self._memory_store[session_id]
        logger.info(f"清除会话: {session_id}")


session_manager = SessionContextManager()
