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
│   ├── db/                      # 数据库模块
│   │   ├── __init__.py
│   │   └── connection.py        # MySQL 连接管理
│   ├── context/                 # 上下文管理
│   │   ├── __init__.py
│   │   ├── session.py           # 会话上下文
│   │   ├── conversation.py      # 对话上下文
│   │   ├── tool_call.py         # 工具调用记录
│   │   └── task.py              # 任务上下文
│   ├── rag.py                   # RAG 数据导入
│   ├── sql.py                   # 数据库操作
│   └── rag_sql.py               # RAG+SQL 混合查询
├── mcp/                         # MCP 服务
│   ├── mcp_server.py            # MCP 服务器 (stdio 模式)
│   ├── api_server.py            # FastAPI 服务 (HTTP 模式)
│   └── llm_client.py            # LLM 调用示例
├── sql/                         # SQL 脚本
│   └── init_database.sql        # 数据库初始化脚本
├── tests/                       # 测试代码
├── rag/                         # RAG 知识库数据
├── tasks/                       # 任务上下文存储
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

## 上下文管理

项目提供三种类型的上下文管理，分别用于不同的业务场景。

### 1. 初始化数据库

在首次使用前，需要初始化 MySQL 数据库：

```bash
# 执行 SQL 脚本创建数据库和表
mysql -u root -p < sql/init_database.sql

# 或在 MySQL 客户端中执行
source sql/init_database.sql
```

SQL 脚本位置：`sql/init_database.sql`

### 2. 环境配置

在 `.env` 文件中添加 MySQL 密码：

```bash
DASHSCOPE_API_KEY=your_api_key_here
MYSQL_PASSWORD=your_mysql_password
```

### 3. 上下文类型说明

| 类型 | 存储方式 | 用途 | 生命周期 |
|------|----------|------|----------|
| 会话上下文 | MySQL | 用户会话管理 | 短期，超时后标记关闭 |
| 对话上下文 | MySQL | 多轮对话历史 | 长期保存 |
| 任务上下文 | 文件 (JSON/MD) | 完整任务流程，含图片 | 长期保存，可生成报告 |

### 4. 代码示例

#### 会话管理

```python
from src.context import get_session_manager

session_mgr = get_session_manager()

# 创建会话
session_id = session_mgr.create_session(
    user_id="user123",
    metadata={"source": "web"}
)

# 获取会话
session = session_mgr.get_session(session_id)

# 更新会话
session_mgr.update_session(session_id, metadata={"last_action": "infer"})

# 关闭会话
session_mgr.close_session(session_id)

# 获取用户的所有活跃会话
sessions = session_mgr.get_active_sessions(user_id="user123")
```

#### 对话管理

```python
from src.context import get_conversation_manager

conv_mgr = get_conversation_manager()

# 添加用户消息
conv_mgr.add_user_message(session_id, "帮我分析这张病理图片")

# 添加助手消息
conv_mgr.add_assistant_message(session_id, "好的，请提供图片路径")

# 添加工具返回消息
conv_mgr.add_tool_message(
    session_id,
    tool_name="infer_medical_image",
    content="图像分析完成：FSGS改变"
)

# 获取对话历史（格式化）
history = conv_mgr.get_history_formatted(session_id)

# 获取最近 N 条消息
messages = conv_mgr.get_last_n_messages(session_id, n=10)

# 清空对话历史
conv_mgr.clear_history(session_id)
```

#### 工具调用记录

```python
from src.context import get_tool_call_recorder

recorder = get_tool_call_recorder()

# 记录工具调用
recorder.record_tool_execution(
    session_id=session_id,
    tool_name="infer_medical_image",
    arguments={"image_path": "./1.png"},
    result="FSGS改变",
    status="success"
)

# 获取工具调用统计
stats = recorder.get_tool_statistics(session_id)

# 获取会话的工具调用记录
calls = recorder.get_calls_by_session(session_id)
```

#### 任务管理

```python
from src.context import get_task_manager

task_mgr = get_task_manager()

# 创建任务
task = task_mgr.create_task(metadata={
    "type": "病理分析",
    "patient_id": "P001"
})
task_id = task.task_id

# 添加执行步骤
task.add_step("image_receive", {"image": "1.png", "size": "2MB"})
task.add_step("infer", {"result": "FSGS", "confidence": 0.95})
task.add_step("rag_search", {"query": "FSGS治疗方案", "results": 5})

# 添加结果文件
task.add_result("diagnosis.json", '{"diagnosis": "FSGS", "recommendation": "激素治疗"}')

# 追加历史记录
task.append_history("## 额外说明\n\n患者有高血压病史。\n")

# 生成完整报告
report_path = task.generate_report()

# 获取任务列表
tasks = task_mgr.list_tasks(status="created")
```

### 5. 配置项

在 `config/config.yaml` 中可以配置上下文管理参数：

```yaml
# MySQL 数据库配置
database:
  mysql:
    host: "localhost"
    port: 3306
    user: "root"
    password: "${MYSQL_PASSWORD}"
    database: "kidney_agent"

# 上下文管理配置
context:
  session_timeout: 3600       # 会话超时时间（秒），默认1小时
  max_history: 50           # 最大保留消息数
  task_dir: "./tasks"       # 任务文件存储目录
  task_retention_days: 90   # 任务保留天数
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
