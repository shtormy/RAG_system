import redis
import json
from typing import List, Optional
import os

SESSION_TTL = 60 * 30  # 30 минут

class SessionManager:
    def __init__(self, host: str = None, port: int = None, db: int = 0):
        host = host or os.getenv('REDIS_HOST', 'localhost')
        port = port or int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def _get_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def get_history(self, session_id: str) -> List[str]:
        key = self._get_key(session_id)
        history = self.redis.get(key)
        if history:
            return json.loads(history)
        return []

    def add_message(self, session_id: str, message: str) -> None:
        key = self._get_key(session_id)
        history = self.get_history(session_id)
        history.append(message)
        self.redis.set(key, json.dumps(history), ex=SESSION_TTL)

    def clear_session(self, session_id: str) -> None:
        key = self._get_key(session_id)
        self.redis.delete(key)

    def refresh_ttl(self, session_id: str) -> None:
        key = self._get_key(session_id)
        self.redis.expire(key, SESSION_TTL) 