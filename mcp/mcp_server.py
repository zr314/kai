"""
MCP 服务器
基于配置和注册中心的现代化 MCP 服务
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 初始化配置和日志
from config import init_logging, get_logger, get, request_context, set_request_id
init_logging()
logger = get_logger(__name__)

# 导入配置
from config import get as config_get

# 延迟导入依赖，避免初始化失败
from pymilvus import connections, Collection
import dashscope
from dashscope import TextEmbedding
from mcp.server.stdio import stdio_server
from mcp.server import Server


# ====================== Milvus 连接 ======================

def init_milvus():
    """连接 Milvus 并加载集合"""
    try:
        milvus_config = config_get('milvus', {})
        connections.connect(
            alias="default",
            host=milvus_config.get('host', 'localhost'),
            port=milvus_config.get('port', 19530)
        )
        collection = Collection(milvus_config.get('collection', 'kidney_rag_collection'))
        collection.load()
        logger.info("Milvus 连接成功")
        return collection
    except Exception as e:
        logger.error(f"Milvus 初始化失败: {e}")
        return None


def get_query_embedding(query_text: str):
    """生成查询文本的向量"""
    try:
        dashscope_config = config_get('dashscope', {})
        milvus_config = config_get('milvus', {})

        response = TextEmbedding.call(
            model=dashscope_config.get('embedding_model', 'text-embedding-v4'),
            input=query_text,
            text_type="query",
            output_format="float"
        )
        if response.status_code == 200:
            embedding = response.output['embeddings'][0]['embedding']
            dimension = milvus_config.get('dimension', 1024)
            return embedding if len(embedding) == dimension else None
    except Exception as e:
        logger.error(f"生成查询向量出错: {e}")
    return None


def search_knowledge_base(query_text: str, top_k: int = 10):
    """RAG 知识库检索"""
    milvus_config = config_get('milvus', {})

    collection = init_milvus()
    if collection is None:
        return "Milvus 连接失败"

    query_embedding = get_query_embedding(query_text)
    if query_embedding is None:
        return "查询向量生成失败"

    search_params = {
        "metric_type": milvus_config.get('metric_type', 'COSINE'),
        "params": {"nprobe": milvus_config.get('nprobe', 50)}
    }

    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source", "text_length"]
        )

        if results and len(results[0]) > 0:
            result_text = f"[Result] 找到 {len(results[0])} 条相关结果:\n\n"
            for i, hit in enumerate(results[0], 1):
                text = hit.entity.get("text", "unknown")
                source = hit.entity.get("source", "unknown")
                result_text += f"[{i}] 相似度: {hit.score:.4f}\n"
                result_text += f"内容: {text[:500]}...\n" if len(text) > 500 else f"内容: {text}\n"
                result_text += f"来源: {source}\n"
                result_text += "-" * 50 + "\n"
            return result_text
        else:
            return "[Error] 未找到相关数据"
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return f"[Error] 检索失败: {e}"


# ====================== MCP 服务器 ======================

app = Server("qwen3vl-glomerulus-medical-infer")


@app.list_tools()
async def list_tools():
    """列出可用的工具"""
    # 从配置读取工具列表
    tools_config = config_get('mcp.tools', [])

    tools = []
    for tool_config in tools_config:
        if not tool_config.get('enabled', True):
            continue

        name = tool_config.get('name')
        description = tool_config.get('description', '')
        params = tool_config.get('params', {})

        # 构建输入 schema
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        if name == "infer_medical_image":
            input_schema["properties"]["image_path"] = {
                "type": "string",
                "description": "图片的本地相对路径，例如: ./1.png"
            }
            input_schema["required"] = ["image_path"]

        elif name == "search_knowledge_base":
            input_schema["properties"]["query_text"] = {
                "type": "string",
                "description": "查询文本，用英文或中文描述要查询的医学问题"
            }
            input_schema["properties"]["top_k"] = {
                "type": "integer",
                "description": "返回结果的数量，默认10条",
                "default": params.get('top_k', 10)
            }
            input_schema["required"] = ["query_text"]

        tools.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema
        })

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """调用工具"""
    # 生成请求 ID 用于日志追踪
    req_id = set_request_id()

    with request_context(req_id):
        logger.info(f"收到工具调用: {name}, 参数: {arguments}")

        try:
            if name == "infer_medical_image":
                # 延迟导入避免启动时加载模型
                from src import model_infer
                image_path = arguments["image_path"]
                result = model_infer.infer(image_path)
                logger.info(f"图像推理完成: {name}")
                return result

            elif name == "search_knowledge_base":
                query_text = arguments["query_text"]
                top_k = arguments.get("top_k", config_get('mcp.tools', [{}])[0].get('params', {}).get('top_k', 10))
                result = search_knowledge_base(query_text, top_k)
                logger.info(f"RAG 检索完成: {name}")
                return result

            else:
                raise ValueError(f"未知工具: {name}")

        except Exception as e:
            logger.error(f"工具调用失败: {name}, error: {e}", exc_info=True)
            raise


# ====================== 主函数 ======================

if __name__ == "__main__":
    import asyncio

    logger.info("启动 MCP 服务器...")
    logger.info(f"环境: {config_get('app.environment', 'unknown')}")
    logger.info(f"日志级别: {config_get('logging.level', 'INFO')}")

    # 初始化 Milvus
    init_milvus()

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )

    asyncio.run(main())
