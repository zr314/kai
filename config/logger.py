"""
日志系统
支持：
- 结构化 JSON 日志
- 文件轮转
- 请求追踪 (request_id)
- 多模块日志级别控制
"""
import os
import sys
import json
import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from .loader import get


class JsonFormatter(logging.Formatter):
    """JSON 格式化器"""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # 添加 request_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        # 添加额外字段
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in [
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'message', 'pathname', 'process', 'processName', 'relativeCreated',
                    'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                    'request_id'
                ]
            }
            if extra_fields:
                log_data['extra'] = extra_fields

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本格式化器"""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging() -> logging.Logger:
    """
    设置日志系统
    从配置读取日志配置
    """
    # 获取日志配置
    log_config = get('logging', {})

    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', 'text')

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 设置格式化器
    if log_format == 'json':
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter()

    # 控制台处理器
    console_config = log_config.get('console', {})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_config.get('level', 'INFO')))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器
    file_config = log_config.get('file', {})
    if file_config.get('enabled', True):
        log_file = file_config.get('path', './logs/app.log')
        log_dir = Path(log_file).parent

        # 创建日志目录
        log_dir.mkdir(parents=True, exist_ok=True)

        # 轮转文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=file_config.get('max_bytes', 10 * 1024 * 1024),
            backupCount=file_config.get('backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# ==================== 请求追踪 ====================

class RequestContextFilter(logging.Filter):
    """请求上下文过滤器"""

    def __init__(self):
        super().__init__()
        self.request_id = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id or 'N/A'
        return True


# 全局请求上下文
_request_context: Dict[str, str] = {}


def set_request_id(request_id: Optional[str] = None) -> str:
    """设置当前请求 ID"""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    _request_context['request_id'] = request_id
    return request_id


def get_request_id() -> str:
    """获取当前请求 ID"""
    return _request_context.get('request_id', 'N/A')


def clear_request_id() -> None:
    """清除请求 ID"""
    _request_context.clear()


@contextmanager
def request_context(request_id: Optional[str] = None):
    """
    请求上下文管理器
    用法:
        with request_context():
            logger.info("xxx")
    """
    req_id = set_request_id(request_id)
    try:
        yield req_id
    finally:
        clear_request_id()


# ==================== 便捷日志函数 ====================

def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


# ==================== 初始化 ====================

def init():
    """初始化日志系统"""
    setup_logging()
    logger = get_logger(__name__)
    logger.info("日志系统初始化完成")
