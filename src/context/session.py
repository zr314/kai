"""
会话上下文管理
管理用户会话的创建、查询、更新和清理
"""
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

from src.db.connection import get_db
from config import get

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.db = get_db()
        self.session_timeout = get('context.session_timeout', 3600)  # 默认1小时

    def create_session(self, user_id: Optional[str] = None, metadata: Dict = None) -> str:
        """
        创建新会话

        Args:
            user_id: 用户ID（可选）
            metadata: 额外元数据

        Returns:
            session_id
        """
        session_id = str(uuid.uuid4())

        sql = """
            INSERT INTO sessions (session_id, user_id, metadata)
            VALUES (%s, %s, %s)
        """
        self.db.execute(sql, (session_id, user_id, json.dumps(metadata) if metadata else None))

        logger.info(f"创建会话: {session_id}, user_id: {user_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        sql = "SELECT * FROM sessions WHERE session_id = %s"
        return self.db.query(sql, (session_id,))

    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        更新会话

        Args:
            session_id: 会话ID
            **kwargs: 要更新的字段
        """
        if not kwargs:
            return False

        # 特殊处理 metadata 字段
        if 'metadata' in kwargs:
            kwargs['metadata'] = json.dumps(kwargs['metadata'])

        # 更新 last_active_at
        kwargs['last_active_at'] = datetime.now()

        set_clause = ', '.join([f"{k} = %s" for k in kwargs.keys()])
        sql = f"UPDATE sessions SET {set_clause} WHERE session_id = %s"

        values = list(kwargs.values()) + [session_id]
        self.db.execute(sql, tuple(values))

        logger.debug(f"更新会话: {session_id}, fields: {list(kwargs.keys())}")
        return True

    def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        return self.update_session(session_id, status='closed')

    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        获取活跃会话

        Args:
            user_id: 用户ID（可选）
        """
        if user_id:
            sql = """
                SELECT * FROM sessions
                WHERE user_id = %s AND status = 'active'
                ORDER BY last_active_at DESC
            """
            return self.db.query_all(sql, (user_id,))
        else:
            sql = """
                SELECT * FROM sessions
                WHERE status = 'active'
                ORDER BY last_active_at DESC
            """
            return self.db.query_all(sql)

    def delete_session(self, session_id: str) -> bool:
        """删除会话（级联删除消息和工具调用）"""
        # 先关闭会话
        self.close_session(session_id)
        logger.info(f"删除会话: {session_id}")
        return True

    def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            清理的会话数量
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(seconds=self.session_timeout)

        sql = """
            UPDATE sessions
            SET status = 'closed'
            WHERE status = 'active' AND last_active_at < %s
        """
        affected = self.db.execute(sql, (cutoff,))

        if affected > 0:
            logger.info(f"清理了 {affected} 个过期会话")

        return affected


# 全局会话管理器
_session_manager = None


def get_session_manager() -> SessionManager:
    """获取会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
