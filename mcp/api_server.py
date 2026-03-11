"""
FastAPI 服务
基于配置和日志系统的现代化 API 服务
"""
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 初始化配置和日志
from config import init_logging, get_logger, get, request_context, set_request_id
init_logging()
logger = get_logger(__name__)

# 导入配置
from config import get as config_get

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# ====================== FastAPI 应用 ======================

app = FastAPI(
    title=config_get('app.name', 'Kidney Medical Agent'),
    description="肾小球病理分析 MCP 工具 API",
    version=config_get('app.version', '1.0.0')
)


# ====================== 请求模型 ======================

class RagQueryRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = None


class ImageInferRequest(BaseModel):
    image_path: str


# ====================== 中间件 ======================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    req_id = set_request_id()

    with request_context(req_id):
        logger.info(f"收到请求: {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            logger.info(f"请求完成: {request.method} {request.url.path}, 状态码: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"请求处理失败: {e}", exc_info=True)
            raise


# ====================== RAG 功能 ======================

def init_milvus():
    """连接 Milvus"""
    try:
        milvus_config = config_get('milvus', {})
        from pymilvus import connections, Collection
        connections.connect(
            alias="default",
            host=milvus_config.get('host', 'localhost'),
            port=milvus_config.get('port', 19530)
        )
        collection = Collection(milvus_config.get('collection', 'kidney_rag_collection'))
        collection.load()
        return collection
    except Exception as e:
        logger.error(f"Milvus 初始化失败: {e}")
        return None


def get_query_embedding(query_text: str):
    """生成查询向量"""
    try:
        dashscope_config = config_get('dashscope', {})
        milvus_config = config_get('milvus', {})

        import dashscope
        from dashscope import TextEmbedding
        dashscope.api_key = dashscope_config.get('api_key', '')

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


# ====================== API 路由 ======================

@app.get("/")
def root():
    """根路径"""
    return {
        "name": config_get('app.name'),
        "version": config_get('app.version'),
        "environment": config_get('app.environment'),
        "message": "肾小球医学 MCP 工具 API"
    }


@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "environment": config_get('app.environment')
    }


@app.get("/tools")
def list_tools():
    """列出所有可用工具"""
    tools_config = config_get('mcp.tools', [])

    tools = []
    for tool_config in tools_config:
        if not tool_config.get('enabled', True):
            continue

        name = tool_config.get('name')
        description = tool_config.get('description', '')
        params = tool_config.get('params', {})

        tools.append({
            "name": name,
            "description": description,
            "params": params
        })

    return {"tools": tools}


@app.get("/stats")
def get_stats():
    """获取工具调用统计"""
    try:
        from src.registry import registry
        return registry.get_stats()
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return {"error": str(e)}


@app.post("/search-knowledge-base")
def search_knowledge_base(req: RagQueryRequest):
    """RAG 知识库检索"""
    milvus_config = config_get('milvus', {})

    # 获取 top_k
    top_k = req.top_k
    if top_k is None:
        for tool in config_get('mcp.tools', []):
            if tool.get('name') == 'search_knowledge_base':
                top_k = tool.get('params', {}).get('top_k', 10)
                break
        if top_k is None:
            top_k = 10

    collection = init_milvus()
    if collection is None:
        raise HTTPException(status_code=500, detail="Milvus 连接失败")

    query_embedding = get_query_embedding(req.query_text)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="查询向量生成失败")

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
            result_data = []
            for i, hit in enumerate(results[0], 1):
                result_data.append({
                    "rank": i,
                    "similarity": round(hit.score, 4),
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "text_length": hit.entity.get("text_length", 0)
                })

            return {
                "success": True,
                "query": req.query_text,
                "total_results": len(result_data),
                "results": result_data
            }
        else:
            return {
                "success": True,
                "query": req.query_text,
                "total_results": 0,
                "results": [],
                "message": "未找到相关数据"
            }
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.post("/infer-medical-image")
def infer_medical_image(req: ImageInferRequest):
    """医学图像推理"""
    try:
        from src import model_infer
        result = model_infer.infer(req.image_path)
        return {
            "success": True,
            "image_path": req.image_path,
            "result": result
        }
    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


# ====================== 主函数 ======================

if __name__ == "__main__":
    server_config = config_get('server', {})

    print("=" * 50)
    print(f"{config_get('app.name')} API")
    print(f"版本: {config_get('app.version')}")
    print(f"环境: {config_get('app.environment')}")
    print(f"日志级别: {config_get('logging.level')}")
    print(f"地址: http://{server_config.get('host', '0.0.0.0')}:{server_config.get('port', 8001)}")
    print(f"API 文档: http://{server_config.get('host', '0.0.0.0')}:{server_config.get('port', 8001)}/docs")
    print("=" * 50)

    uvicorn.run(
        app,
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8001),
        reload=server_config.get('reload', False)
    )
