"""
MySQL 数据库连接和操作
"""
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import logging

from config import get

logger = logging.getLogger(__name__)


class Database:
    """数据库连接管理器"""

    _instance = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._connection is None:
            self._connect()

    def _connect(self):
        """建立数据库连接"""
        try:
            self._connection = pymysql.connect(
                host=get('database.mysql.host', 'localhost'),
                port=get('database.mysql.port', 3306),
                user=get('database.mysql.user', 'root'),
                password=get('database.mysql.password', ''),
                database=get('database.mysql.database', 'kidney_agent'),
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=True
            )
            logger.info("MySQL 连接成功")
        except Exception as e:
            logger.error(f"MySQL 连接失败: {e}")
            raise

    @property
    def connection(self):
        """获取连接，自动重连"""
        if self._connection is None or not self._connection.open:
            self._connect()
        return self._connection

    @contextmanager
    def cursor(self):
        """获取游标的上下文管理器"""
        cursor = self.connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute(self, sql: str, args: tuple = None) -> int:
        """执行 SQL"""
        with self.cursor() as cursor:
            return cursor.execute(sql, args)

    def query(self, sql: str, args: tuple = None) -> List[Dict]:
        """查询单条"""
        with self.cursor() as cursor:
            cursor.execute(sql, args)
            return cursor.fetchone()

    def query_all(self, sql: str, args: tuple = None) -> List[Dict]:
        """查询多条"""
        with self.cursor() as cursor:
            cursor.execute(sql, args)
            return cursor.fetchall()

    def close(self):
        """关闭连接"""
        if self._connection and self._connection.open:
            self._connection.close()
            logger.info("MySQL 连接已关闭")


# 全局数据库实例
db = Database()


def get_db() -> Database:
    """获取数据库实例"""
    return db
