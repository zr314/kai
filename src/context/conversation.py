"""
对话上下文管理
管理会话中的消息历史
"""
import json
from typing import Optional, List, Dict, Any
import logging

from src.db.connection import get_db
from config import get

logger = logging.getLogger(__name__)


class ConversationManager:
    """对话历史管理器"""

    def __init__(self):
        self.db = get_db()
        self.max_history = get('context.max_history', 50)  # 最大保留消息数

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: Optional[str] = None
    ) -> int:
        """
        添加消息

        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system/tool)
            content: 消息内容
            tool_name: 工具名称（如果是工具调用）

        Returns:
            message_id
        """
        sql = """
            INSERT INTO messages (session_id, role, content, tool_name)
            VALUES (%s, %s, %s, %s)
        """
        cursor = self.db.connection.cursor()
        cursor.execute(sql, (session_id, role, content, tool_name))
        message_id = cursor.lastrowid
        cursor.close()

        logger.debug(f"添加消息: session={session_id}, role={role}, message_id={message_id}")
        return message_id

    def add_user_message(self, session_id: str, content: str) -> int:
        """添加用户消息"""
        return self.add_message(session_id, 'user', content)

    def add_assistant_message(self, session_id: str, content: str) -> int:
        """添加助手消息"""
        return self.add_message(session_id, 'assistant', content)

    def add_tool_message(
        self,
        session_id: str,
        tool_name: str,
        content: str
    ) -> int:
        """添加工具返回消息"""
        return self.add_message(session_id, 'tool', content, tool_name)

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            limit: 返回数量限制
            offset: 偏移量
        """
        if limit is None:
            limit = self.max_history

        sql = """
            SELECT message_id, role, content, tool_name, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s OFFSET %s
        """
        return self.db.query_all(sql, (session_id, limit, offset))

    def get_history_formatted(self, session_id: str, limit: Optional[int] = None) -> str:
        """
        获取格式化的对话历史（用于上下文）

        Returns:
            格式化后的历史字符串
        """
        messages = self.get_history(session_id, limit)

        if not messages:
            return ""

        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'tool':
                formatted.append(f"[Tool: {msg['tool_name']}] {content}")
            else:
                formatted.append(f"[{role.upper()}] {content}")

        return "\n".join(formatted)

    def get_last_n_messages(self, session_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近 N 条消息"""
        sql = """
            SELECT message_id, role, content, tool_name, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        messages = self.db.query_all(sql, (session_id, n))
        # 反转顺序，保持时间正序
        return list(reversed(messages))

    def clear_history(self, session_id: str) -> int:
        """清空会话历史"""
        sql = "DELETE FROM messages WHERE session_id = %s"
        affected = self.db.execute(sql, (session_id,))
        logger.info(f"清空会话历史: {session_id}, 删除了 {affected} 条消息")
        return affected

    def count_messages(self, session_id: str) -> int:
        """统计会话消息数"""
        sql = "SELECT COUNT(*) as cnt FROM messages WHERE session_id = %s"
        result = self.db.query(sql, (session_id,))
        return result['cnt'] if result else 0


# 全局对话管理器
_conversation_manager = None


def get_conversation_manager() -> ConversationManager:
    """获取对话管理器实例"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
