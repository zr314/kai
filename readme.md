# 肾小球医学 Agent

基于 Qwen3-VL 的肾小球病理分析 MCP 工具集，提供医学图像推理和 RAG 知识库检索功能。

## 目录结构

```
agent/
├── config/                      # 配置和日志系统
│   ├── __init__.py
│   ├── config.yaml              # 主配置文件
│   ├── config_dev.yaml          # 开发环境配置
│   ├── config_prod.yaml         # 生产环境配置
│   ├── loader.py                # 配置加载器
│   └── logger.py                # 日志系统
├── src/                         # 核心源代码
│   ├── __init__.py
│   ├── model_infer.py           # 图像推理模块
│   ├── registry.py              # MCP 工具注册中心
│   ├── rag.py                   # RAG 数据导入
│   ├── sql.py                   # 数据库操作
│   └── rag_sql.py               # RAG+SQL 混合查询
├── mcp/                         # MCP 服务
│   ├── mcp_server.py            # MCP 服务器 (stdio 模式)
│   ├── api_server.py            # FastAPI 服务 (HTTP 模式)
│   └── llm_client.py            # LLM 调用示例
├── tests/                       # 测试代码
├── rag/                         # RAG 知识库数据
├── Qwen3-VL-4B-Instruct/        # 基础模型
├── qwen3vl-4b-medical-lora-sft/ # 微调模型
└── 1.png                        # 测试图片
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件：

```bash
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 启动服务

**MCP 服务器 (stdio 模式):**
```bash
conda activate qwen3vl
python mcp/mcp_server.py
```

**HTTP API 服务:**
```bash
conda activate qwen3vl
python mcp/api_server.py
```

服务地址: http://localhost:8001
API 文档: http://localhost:8001/docs

## 配置系统

### 配置文件结构

配置文件位于 `config/` 目录：

- `config.yaml` - 主配置文件
- `config_dev.yaml` - 开发环境覆盖配置
- `config_prod.yaml` - 生产环境覆盖配置

### 配置项说明

```yaml
# 应用配置
app:
  name: "kidney-medical-agent"
  version: "1.0.0"
  environment: "dev"  # dev | prod

# 日志配置
logging:
  level: "DEBUG"     # DEBUG | INFO | WARNING | ERROR
  format: "json"     # json | text
  file:
    enabled: true
    path: "./logs/app.log"
  console:
    enabled: true

# Milvus 向量数据库
milvus:
  host: "localhost"
  port: 19530
  collection: "kidney_rag_collection"

# 阿里云百炼
dashscope:
  api_key: "${DASHSCOPE_API_KEY}"  # 从环境变量读取

# MCP 工具配置
mcp:
  tools:
    - name: "infer_medical_image"
      enabled: true
    - name: "search_knowledge_base"
      enabled: true
```

### 使用配置

```python
from config import get, get_config

# 获取配置值
host = get('milvus.host')
port = get('milvus.port')

# 获取完整配置
config = get_config()
```

## 日志系统

### 初始化

```python
from config import init_logging, get_logger
init_logging()

logger = get_logger(__name__)
logger.info("消息")
```

### 请求追踪

```python
from config import request_context, set_request_id

# 使用上下文管理器
with request_context():
    logger.info("请求处理中")

# 手动设置
req_id = set_request_id()
```

### 日志格式

支持 JSON 和文本格式：

**JSON 格式:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "mcp_server",
  "message": "收到请求",
  "request_id": "abc123"
}
```

## MCP 工具

### infer_medical_image
- **功能**: 对肾小球病理图片进行诊断
- **参数**: `{"image_path": "./1.png"}`

### search_knowledge_base
- **功能**: RAG 知识库检索
- **参数**: `{"query_text": "crescentic glomerulonephritis", "top_k": 10}`

## API 调用示例

### RAG 知识库检索
```bash
curl -X POST http://localhost:8001/search-knowledge-base \
  -H "Content-Type: application/json" \
  -d '{"query_text": "crescentic glomerulonephritis", "top_k": 5}'
```

### 医学图像推理
```bash
curl -X POST http://localhost:8001/infer-medical-image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "./1.png"}'
```

## 添加新 MCP 工具

### 1. 在配置中注册

编辑 `config/config.yaml`:

```yaml
mcp:
  tools:
    - name: "my_new_tool"
      enabled: true
      description: "新工具描述"
      params:
        default_param: "value"
```

### 2. 在 mcp_server.py 中实现

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_new_tool":
        # 实现逻辑
        return result
```

### 3. 使用装饰器注册 (可选)

```python
from src import register_tool

@register_tool("my_tool", description="我的工具")
def my_tool(param1: str):
    return param1
```

## Claude Code 配置

在 Claude Code 配置文件中添加：

```json
{
  "mcpServers": {
    "qwen3vl-medical": {
      "command": "conda",
      "args": ["run", "-n", "qwen3vl", "python", "D:/agent/mcp/mcp_server.py"]
    }
  }
}
```

## 环境要求

- Python 3.9+
- conda 环境: qwen3vl
- 依赖: 见 requirements.txt
