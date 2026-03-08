"""
LLM调用MCP工具的示例
让LLM直接执行这个脚本来调用API
"""
import requests
import json

API_BASE = "http://localhost:8001"

def call_rag(query: str, top_k: int = 5):
    """调用RAG知识库检索"""
    resp = requests.post(
        f"{API_BASE}/search-knowledge-base",
        json={"query_text": query, "top_k": top_k}
    )
    return resp.json()

def call_image_infer(image_path: str):
    """调用医学图像推理"""
    resp = requests.post(
        f"{API_BASE}/infer-medical-image",
        json={"image_path": image_path}
    )
    return resp.json()

# 示例调用
if __name__ == "__main__":
    # RAG查询
    result = call_rag("crescentic glomerulonephritis pathological structure")
    print("RAG结果:", json.dumps(result, indent=2, ensure_ascii=False)[:500])

    # 图像推理
    # result = call_image_infer("./1.png")
    # print("图像推理:", result)
