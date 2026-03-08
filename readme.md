# 肾小球医学MCP工具

基于Qwen3-VL的肾小球病理分析MCP工具集，提供医学图像推理和RAG知识库检索功能。

## 目录结构

```
agent/
├── src/                    # 核心源代码
│   ├── model_infer.py      # 图像推理模块
│   ├── rag.py              # RAG数据导入
│   ├── sql.py              # 数据库操作
│   └── rag_sql.py          # RAG+SQL混合查询
├── mcp/                    # MCP服务
│   ├── mcp_server.py       # MCP服务器 (stdio模式)
│   ├── api_server.py       # FastAPI服务 (HTTP模式)
│   └── llm_client.py       # LLM调用示例
├── tests/                  # 测试代码
│   ├── test.py             # RAG查询测试
│   └── ragtest.py          # RAG功能测试
├── rag/                    # RAG知识库数据
├── Qwen3-VL-4B-Instruct/   # 基础模型
├── qwen3vl-4b-medical-lora-sft/  # 微调模型
└── 1.png                   # 测试图片
```

## 快速开始

### 1. 启动MCP服务器 (stdio模式)

```bash
conda activate qwen3vl
python mcp/mcp_server.py
```

### 2. 启动API服务 (HTTP模式)

```bash
conda activate qwen3vl
python mcp/api_server.py
```

服务地址: http://localhost:8001
API文档: http://localhost:8001/docs

## MCP工具

### infer_medical_image
- **功能**: 对肾小球病理图片进行诊断
- **参数**: `{"image_path": "./1.png"}`

### search_knowledge_base
- **功能**: RAG知识库检索
- **参数**: `{"query_text": "crescentic glomerulonephritis", "top_k": 10}`

## API调用示例

### RAG知识库检索
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

## Claude Code配置

在Claude Code配置文件中添加：

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
- conda环境: qwen3vl
- 依赖: pymilvus, dashscope, fastapi, uvicorn, transformers, torch

