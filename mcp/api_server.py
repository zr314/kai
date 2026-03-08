"""
FastAPI服务 - 包装MCP工具为REST API
让LLM可以通过HTTP调用RAG查询和医学图像推理工具
"""
import os
import json
from dotenv import load_dotenv
from pymilvus import connections, Collection
import dashscope
from dashscope import TextEmbedding
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import model_infer

# ====================== 配置项 ======================
load_dotenv()

# 1. Milvus配置
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

# ====================== FastAPI应用 ======================
app = FastAPI(
    title="肾小球医学MCP工具API",
    description="提供RAG知识库检索和医学图像推理功能",
    version="1.0.0"
)

# ====================== 请求模型 ======================
class RagQueryRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 10

class ImageInferRequest(BaseModel):
    image_path: str

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

# ====================== API路由 ======================
@app.get("/")
def root():
    return {
        "message": "肾小球医学MCP工具API",
        "tools": [
            "/search-knowledge-base - RAG知识库检索",
            "/infer-medical-image - 医学图像推理"
        ]
    }

@app.get("/tools")
def list_tools():
    """列出所有可用工具"""
    return {
        "tools": [
            {
                "name": "search_knowledge_base",
                "description": "在医学知识库中进行RAG检索。输入查询文本，返回相关的医学知识内容。适用于询问肾小球疾病相关的医学问题。",
                "endpoint": "/search-knowledge-base",
                "method": "POST",
                "params": {
                    "query_text": "string (必需) - 查询文本",
                    "top_k": "integer (可选) - 返回结果数量，默认10"
                }
            },
            {
                "name": "infer_medical_image",
                "description": "对肾小球病理图片进行诊断。输入一张本地肾小球图片路径，模型会输出诊断结果报告。",
                "endpoint": "/infer-medical-image",
                "method": "POST",
                "params": {
                    "image_path": "string (必需) - 图片的本地相对路径"
                }
            }
        ]
    }

@app.post("/search-knowledge-base")
def search_knowledge_base(req: RagQueryRequest):
    """RAG知识库检索"""
    collection = init_milvus()
    if collection is None:
        raise HTTPException(status_code=500, detail="Milvus connection failed")

    # 生成查询向量
    query_embedding = get_query_embedding(req.query_text)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Query embedding generation failed")

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
            limit=req.top_k,
            output_fields=["text", "source", "text_length"]
        )

        if results and len(results[0]) > 0:
            result_data = []
            for i, hit in enumerate(results[0], 1):
                try:
                    text = hit.entity.get("text")
                except:
                    text = "unknown"
                try:
                    source = hit.entity.get("source")
                except:
                    source = "unknown"
                try:
                    text_length = hit.entity.get("text_length")
                except:
                    text_length = 0

                result_data.append({
                    "rank": i,
                    "similarity": round(hit.score, 4),
                    "text": text,
                    "source": source,
                    "text_length": text_length
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
                "message": "No related data found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/infer-medical-image")
def infer_medical_image(req: ImageInferRequest):
    """医学图像推理"""
    try:
        result = model_infer.infer(req.image_path)
        return {
            "success": True,
            "image_path": req.image_path,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ====================== 主函数 ======================
if __name__ == "__main__":
    print("=" * 50)
    print("Kidney Medical MCP Tools API")
    print("Address: http://localhost:8001")
    print("API Docs: http://localhost:8001/docs")
    print("Tools: http://localhost:8001/tools")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)
