import os
import json
from dotenv import load_dotenv
from pymilvus import connections, Collection
import dashscope
from dashscope import TextEmbedding
from mcp.server.stdio import stdio_server
from mcp.server import Server
import model_infer

# ====================== 配置项 ======================
load_dotenv()

# 1. Milvus配置import os
# import json
# from dotenv import load_dotenv
# from pymilvus import connections, Collection
# import dashscope
# from dashscope import TextEmbedding
# from mcp.server.stdio import stdio_server
# from mcp.server import Server
# import model_infer
#
# # ====================== 配置项 ======================
# load_dotenv()
#
# # 1. Milvus配置
# MILVUS_HOST = "localhost"
# MILVUS_PORT = 19530
# COLLECTION_NAME = "kidney_rag_collection"
# DIMENSION = 1024
#
# # 2. 阿里云百炼配置
#
#
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# EMBEDDING_MODEL = "text-embedding-v4"
#
# # 3. 检索配置
# TOP_K = 10
# METRIC_TYPE = "COSINE"
#
# # 初始化百炼SDK
# dashscope.api_key = DASHSCOPE_API_KEY
#
# # ====================== RAG查询功能 ======================
# def init_milvus():
#     """连接Milvus并加载集合"""
#     try:
#         connections.connect(
#             alias="default",
#             host=MILVUS_HOST,
#             port=MILVUS_PORT
#         )
#         collection = Collection(COLLECTION_NAME)
#         collection.load()
#         return collection
#     except Exception as e:
#         print(f"❌ Milvus初始化失败：{e}")
#         return None
#
# def get_query_embedding(query_text: str) -> list:
#     """生成查询文本的向量"""
#     try:
#         response = TextEmbedding.call(
#             model=EMBEDDING_MODEL,
#             input=query_text,
#             text_type="query",
#             output_format="float"
#         )
#         if response.status_code == 200:
#             embedding = response.output['embeddings'][0]['embedding']
#             return embedding if len(embedding) == DIMENSION else None
#     except Exception as e:
#         print(f"❌ 生成查询向量出错：{e}")
#     return None
#
# def search_knowledge_base(query_text: str, top_k: int = TOP_K):
#     """RAG知识库检索"""
#     collection = init_milvus()
#     if collection is None:
#         return "❌ Milvus连接失败"
#
#     # 生成查询向量
#     query_embedding = get_query_embedding(query_text)
#     if query_embedding is None:
#         return "❌ 查询向量生成失败"
#
#     # 构建检索参数
#     search_params = {
#         "metric_type": METRIC_TYPE,
#         "params": {"nprobe": 50}
#     }
#
#     # 执行检索
#     try:
#         results = collection.search(
#             data=[query_embedding],
#             anns_field="embedding",
#             param=search_params,
#             limit=top_k,
#             output_fields=["text", "source", "text_length"]
#         )
#
#         if results and len(results[0]) > 0:
#             result_text = f"[Result] Found {len(results[0])} related results:\n\n"
#             for i, hit in enumerate(results[0], 1):
#                 try:
#                     text = hit.entity.get("text")
#                 except:
#                     text = "unknown"
#                 try:
#                     source = hit.entity.get("source")
#                 except:
#                     source = "unknown"
#                 result_text += f"[{i}] Similarity: {hit.score:.4f}\n"
#                 result_text += f"Content: {text[:500]}...\n" if len(text) > 500 else f"Content: {text}\n"
#                 result_text += f"Source: {source}\n"
#                 result_text += "-" * 50 + "\n"
#             return result_text
#         else:
#             return "[Error] No related data found"
#     except Exception as e:
#         return f"[Error] Search failed: {e}"
#
#
# app = Server("qwen3vl-glomerulus-medical-infer")
#
#
# @app.list_tools()
# async def list_tools():
#     """列出可用的工具"""
#     return [
#         {
#             "name": "infer_medical_image",
#             "description": "对肾小球病理图片进行诊断。输入一张本地肾小球图片路径，模型会输出诊断结果报告。",
#             "inputSchema": {
#                 "type": "object",
#                 "properties": {
#                     "image_path": {
#                         "type": "string",
#                         "description": "图片的本地相对路径，例如: ./1.png"
#                     }
#                 },
#                 "required": ["image_path"]
#             }
#         },
#         {
#             "name": "search_knowledge_base",
#             "description": "在医学知识库中进行RAG检索。输入查询文本，返回相关的医学知识内容。适用于询问肾小球疾病相关的医学问题。",
#             "inputSchema": {
#                 "type": "object",
#                 "properties": {
#                     "query_text": {
#                         "type": "string",
#                         "description": "查询文本，用英文或中文描述要查询的医学问题，例如: 'crescentic glomerulonephritis pathological structure' 或 '新月体性肾小球肾炎病理结构'"
#                     },
#                     "top_k": {
#                         "type": "integer",
#                         "description": "返回结果的数量，默认10条",
#                         "default": 10
#                     }
#                 },
#                 "required": ["query_text"]
#             }
#         }
#     ]
#
#
# @app.call_tool()
# async def call_tool(name: str, arguments: dict):
#     """调用工具"""
#     if name == "infer_medical_image":
#         image_path = arguments["image_path"]
#         result = model_infer.infer(image_path)
#         return result
#     elif name == "search_knowledge_base":
#         query_text = arguments["query_text"]
#         top_k = arguments.get("top_k", TOP_K)
#         result = search_knowledge_base(query_text, top_k)
#         return result
#     else:
#         raise ValueError(f"Unknown tool: {name}")
#
#
# if __name__ == "__main__":
#     import asyncio
#
#     async def main():
#         async with stdio_server() as (read_stream, write_stream):
#             await app.run(
#                 read_stream,
#                 write_stream,
#                 app.create_initialization_options()
#             )
#
#     asyncio.run(main())
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "kidney_rag_collection"
DIMENSION = 1024

# 2. 阿里云百炼配置


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v4"

# 3. 检索配置
TOP_K = 10
METRIC_TYPE = "COSINE"

# 初始化百炼SDK
dashscope.api_key = DASHSCOPE_API_KEY

# ====================== RAG查询功能 ======================
def init_milvus():
    """连接Milvus并加载集合"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection
    except Exception as e:
        print(f"❌ Milvus初始化失败：{e}")
        return None

def get_query_embedding(query_text: str) -> list:
    """生成查询文本的向量"""
    try:
        response = TextEmbedding.call(
            model=EMBEDDING_MODEL,
            input=query_text,
            text_type="query",
            output_format="float"
        )
        if response.status_code == 200:
            embedding = response.output['embeddings'][0]['embedding']
            return embedding if len(embedding) == DIMENSION else None
    except Exception as e:
        print(f"❌ 生成查询向量出错：{e}")
    return None

def search_knowledge_base(query_text: str, top_k: int = TOP_K):
    """RAG知识库检索"""
    collection = init_milvus()
    if collection is None:
        return "❌ Milvus连接失败"

    # 生成查询向量
    query_embedding = get_query_embedding(query_text)
    if query_embedding is None:
        return "❌ 查询向量生成失败"

    # 构建检索参数
    search_params = {
        "metric_type": METRIC_TYPE,
        "params": {"nprobe": 50}
    }

    # 执行检索
    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source", "text_length"]
        )

        if results and len(results[0]) > 0:
            result_text = f"[Result] Found {len(results[0])} related results:\n\n"
            for i, hit in enumerate(results[0], 1):
                try:
                    text = hit.entity.get("text")
                except:
                    text = "unknown"
                try:
                    source = hit.entity.get("source")
                except:
                    source = "unknown"
                result_text += f"[{i}] Similarity: {hit.score:.4f}\n"
                result_text += f"Content: {text[:500]}...\n" if len(text) > 500 else f"Content: {text}\n"
                result_text += f"Source: {source}\n"
                result_text += "-" * 50 + "\n"
            return result_text
        else:
            return "[Error] No related data found"
    except Exception as e:
        return f"[Error] Search failed: {e}"


app = Server("qwen3vl-glomerulus-medical-infer")


@app.list_tools()
async def list_tools():
    """列出可用的工具"""
    return [
        {
            "name": "infer_medical_image",
            "description": "对肾小球病理图片进行诊断。输入一张本地肾小球图片路径，模型会输出诊断结果报告。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片的本地相对路径，例如: ./1.png"
                    }
                },
                "required": ["image_path"]
            }
        },
        {
            "name": "search_knowledge_base",
            "description": "在医学知识库中进行RAG检索。输入查询文本，返回相关的医学知识内容。适用于询问肾小球疾病相关的医学问题。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "查询文本，用英文或中文描述要查询的医学问题，例如: 'crescentic glomerulonephritis pathological structure' 或 '新月体性肾小球肾炎病理结构'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果的数量，默认10条",
                        "default": 10
                    }
                },
                "required": ["query_text"]
            }
        }
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """调用工具"""
    if name == "infer_medical_image":
        image_path = arguments["image_path"]
        result = model_infer.infer(image_path)
        return result
    elif name == "search_knowledge_base":
        query_text = arguments["query_text"]
        top_k = arguments.get("top_k", TOP_K)
        result = search_knowledge_base(query_text, top_k)
        return result
    else:
        raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    import asyncio

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )

    asyncio.run(main())
