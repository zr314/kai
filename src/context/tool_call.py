"""
工具调用记录管理
记录会话中所有的工具调用
"""
import json
import time
from typing import Optional, Dict, Any, List
import logging

from src.db.connection import get_db

logger = logging.getLogger(__name__)


class ToolCallRecorder:
    """工具调用记录器"""

    def __init__(self):
        self.db = get_db()

    def record_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict,
        message_id: Optional[int] = None
    ) -> int:
        """
        记录工具调用开始

        Args:
            session_id: 会话ID
            tool_name: 工具名称
            arguments: 工具参数
            message_id: 关联的消息ID

        Returns:
            call_id
        """
        sql = """
            INSERT INTO tool_calls (session_id, message_id, tool_name, arguments)
            VALUES (%s, %s, %s, %s)
        """
        cursor = self.db.connection.cursor()
        cursor.execute(sql, (session_id, message_id, tool_name, json.dumps(arguments)))
        call_id = cursor.lastrowid
        cursor.close()

        logger.debug(f"记录工具调用: {tool_name}, call_id={call_id}")
        return call_id

    def update_result(
        self,
        call_id: int,
        result: str,
        status: str = 'success',
        duration_ms: Optional[int] = None
    ) -> bool:
        """
        更新工具调用结果

        Args:
            call_id: 调用ID
            result: 返回结果
            status: 状态 (success/error)
            duration_ms: 耗时（毫秒）
        """
        sql = """
            UPDATE tool_calls
            SET result = %s, status = %s, duration_ms = %s
            WHERE call_id = %s
        """
        self.db.execute(sql, (result, status, duration_ms, call_id))
        logger.debug(f"更新工具调用结果: call_id={call_id}, status={status}")
        return True

    def record_tool_execution(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict,
        result: str,
        message_id: Optional[int] = None,
        status: str = 'success'
    ) -> int:
        """
        记录完整的工具调用（开始+结果）

        Args:
            session_id: 会话ID
            tool_name: 工具名称
            arguments: 工具参数
            result: 返回结果
            message_id: 关联的消息ID
            status: 状态

        Returns:
            call_id
        """
        sql = """
            INSERT INTO tool_calls (session_id, message_id, tool_name, arguments, result, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor = self.db.connection.cursor()
        cursor.execute(sql, (
            session_id,
            message_id,
            tool_name,
            json.dumps(arguments),
            result,
            status
        ))
        call_id = cursor.lastrowid
        cursor.close()

        logger.debug(f"记录工具执行: {tool_name}, call_id={call_id}, status={status}")
        return call_id

    def get_calls_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取会话的工具调用记录"""
        sql = """
            SELECT call_id, tool_name, arguments, result, status, duration_ms, created_at
            FROM tool_calls
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        return self.db.query_all(sql, (session_id, limit))

    def get_tool_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取工具调用统计

        Args:
            session_id: 会话ID（可选）
        """
        if session_id:
            sql = """
                SELECT
                    tool_name,
                    COUNT(*) as call_count,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                    AVG(duration_ms) as avg_duration_ms
                FROM tool_calls
                WHERE session_id = %s
                GROUP BY tool_name
            """
            return self.db.query_all(sql, (session_id,))
        else:
            sql = """
                SELECT
                    tool_name,
                    COUNT(*) as call_count,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                    AVG(duration_ms) as avg_duration_ms
                FROM tool_calls
                GROUP BY tool_name
            """
            return self.db.query_all(sql)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """获取会话的工具调用摘要"""
        sql = """
            SELECT
                COUNT(*) as total_calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_calls,
                AVG(duration_ms) as avg_duration_ms,
                MAX(created_at) as last_call_at
            FROM tool_calls
            WHERE session_id = %s
        """
        return self.db.query(sql, (session_id,))


# 全局记录器
_tool_call_recorder = None


def get_tool_call_recorder() -> ToolCallRecorder:
    """获取工具调用记录器实例"""
    global _tool_call_recorder
    if _tool_call_recorder is None:
        _tool_call_recorder = ToolCallRecorder()
    return _tool_call_recorder
