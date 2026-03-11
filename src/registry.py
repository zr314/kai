"""
MCP 工具注册中心
支持配置驱动的工具加载和动态注册
"""
import logging
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from config import get, get_logger

logger = get_logger(__name__)


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    call_count: int = 0
    error_count: int = 0
    total_time: float = 0.0


class ToolRegistry:
    """
    MCP 工具注册中心
    管理所有可用工具的注册、调用和统计
    """

    _instance = None
    _tools: Dict[str, Callable] = {}
    _metadata: Dict[str, ToolMetadata] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._tools:
            self._load_from_config()

    def _load_from_config(self) -> None:
        """从配置加载工具"""
        try:
            mcp_config = get('mcp.tools', [])
            for tool_config in mcp_config:
                name = tool_config.get('name')
                enabled = tool_config.get('enabled', True)
                description = tool_config.get('description', '')
                params = tool_config.get('params', {})

                self._metadata[name] = ToolMetadata(
                    name=name,
                    description=description,
                    enabled=enabled,
                    params=params
                )

                if enabled:
                    logger.info(f"工具已加载: {name}")
                else:
                    logger.info(f"工具已禁用: {name}")

        except Exception as e:
            logger.error(f"从配置加载工具失败: {e}")

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        注册工具

        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
            params: 默认参数
        """
        self._tools[name] = func

        # 更新或创建元数据
        if name in self._metadata:
            self._metadata[name].enabled = True
        else:
            self._metadata[name] = ToolMetadata(
                name=name,
                description=description,
                enabled=True,
                params=params or {}
            )

        logger.info(f"工具注册成功: {name}")

    def unregister(self, name: str) -> bool:
        """
        注销工具

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            if name in self._metadata:
                self._metadata[name].enabled = False
            logger.info(f"工具已注销: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        tool = self._tools.get(name)
        if tool is None:
            logger.warning(f"工具不存在: {name}")
        return tool

    def call(self, name: str, **kwargs) -> Any:
        """
        调用工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具返回值
        """
        start_time = datetime.now()

        # 获取工具
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"工具未找到: {name}")

        # 检查是否启用
        if name in self._metadata and not self._metadata[name].enabled:
            raise ValueError(f"工具已禁用: {name}")

        try:
            # 合并默认参数
            default_params = {}
            if name in self._metadata:
                default_params = self._metadata[name].params.copy()
            default_params.update(kwargs)

            # 调用工具
            result = tool(**default_params)

            # 更新统计
            self._update_stats(name, error=False, duration=(datetime.now() - start_time).total_seconds())

            logger.info(f"工具调用成功: {name}")
            return result

        except Exception as e:
            # 更新错误统计
            self._update_stats(name, error=True, duration=(datetime.now() - start_time).total_seconds())
            logger.error(f"工具调用失败: {name}, error: {e}")
            raise

    def _update_stats(self, name: str, error: bool, duration: float) -> None:
        """更新工具统计信息"""
        if name in self._metadata:
            meta = self._metadata[name]
            meta.call_count += 1
            if error:
                meta.error_count += 1
            meta.total_time += duration

    def list_tools(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        列出所有工具

        Args:
            enabled_only: 只返回已启用的工具

        Returns:
            工具列表
        """
        tools = []
        for name, meta in self._metadata.items():
            if enabled_only and not meta.enabled:
                continue
            tools.append({
                "name": name,
                "description": meta.description,
                "enabled": meta.enabled,
                "params": meta.params,
                "call_count": meta.call_count,
                "error_count": meta.error_count,
                "avg_time": meta.total_time / meta.call_count if meta.call_count > 0 else 0
            })
        return tools

    def get_stats(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        return {
            name: {
                "call_count": meta.call_count,
                "error_count": meta.error_count,
                "success_count": meta.call_count - meta.error_count,
                "total_time": meta.total_time,
                "avg_time": meta.total_time / meta.call_count if meta.call_count > 0 else 0,
                "error_rate": meta.error_count / meta.call_count if meta.call_count > 0 else 0
            }
            for name, meta in self._metadata.items()
        }

    def reload(self) -> None:
        """重新加载配置"""
        self._tools.clear()
        self._metadata.clear()
        self._load_from_config()
        logger.info("工具注册中心已重新加载")


# 全局工具注册中心实例
registry = ToolRegistry()


def register_tool(name: str, description: str = "", params: Optional[Dict[str, Any]] = None):
    """
    装饰器：注册 MCP 工具

    用法:
        @register_tool("my_tool", description="我的工具")
        def my_tool(param1: str):
            return param1
    """
    def decorator(func: Callable) -> Callable:
        registry.register(name, func, description, params)
        return func
    return decorator
