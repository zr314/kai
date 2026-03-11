"""
配置加载器
支持：
- YAML 配置文件
- 环境变量替换
- 多环境配置覆盖
- 配置验证
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def load(self, env: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置

        Args:
            env: 环境名称 (dev/prod)，默认从配置文件读取
        """
        # 确定环境
        if env is None:
            # 先加载主配置获取环境
            base_dir = Path(__file__).parent
            config_file = base_dir / "config.yaml"

            with open(config_file, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)
                env = base_config.get('app', {}).get('environment', 'dev')

        # 加载基础配置
        base_dir = Path(__file__).parent
        config_file = base_dir / "config.yaml"
        self._config = self._load_yaml(config_file)

        # 加载环境特定配置并合并
        env_config_file = base_dir / f"config_{env}.yaml"
        if env_config_file.exists():
            env_config = self._load_yaml(env_config_file)
            self._merge_config(self._config, env_config)

        # 处理环境变量替换
        self._resolve_env_vars(self._config)

        logger.info(f"配置加载完成 (env={env})")
        return self._config

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载 YAML 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML 解析错误 {file_path}: {e}")
            return {}

    def _merge_config(self, base: Dict, override: Dict) -> None:
        """递归合并配置，override 优先"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _resolve_env_vars(self, config: Any) -> None:
        """递归处理环境变量替换 ${VAR_NAME}"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, '')
                elif isinstance(value, (dict, list)):
                    self._resolve_env_vars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                    env_var = item[2:-1]
                    config[i] = os.getenv(env_var, '')
                elif isinstance(item, (dict, list)):
                    self._resolve_env_vars(item)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号路径
        例如: config.get('milvus.host')
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def reload(self, env: Optional[str] = None) -> Dict[str, Any]:
        """重新加载配置"""
        self._config = {}
        return self.load(env)


# 全局配置实例
config = ConfigLoader()


def get_config() -> Dict[str, Any]:
    """获取配置字典"""
    return config.config


def get(key: str, default: Any = None) -> Any:
    """便捷获取配置"""
    return config.get(key, default)
