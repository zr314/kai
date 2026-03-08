"""
使用LangChain让LLM自动调用MCP工具
需要安装: pip install langchain-openai
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests

API_BASE = "http://localhost:8001"

@tool
def search_knowledge_base(query_text: str, top_k: int = 10):
    """在医学知识库中进行RAG检索。适用于询问肾小球疾病相关的医学问题。"""
    resp = requests.post(
        f"{API_BASE}/search-knowledge-base",
        json={"query_text": query_text, "top_k": top_k}
    )
    return resp.json()

@tool
def infer_medical_image(image_path: str):
    """对肾小球病理图片进行诊断。"""
    resp = requests.post(
        f"{API_BASE}/infer-medical-image",
        json={"image_path": image_path}
    )
    return resp.json()

# 初始化LLM
llm = ChatOpenAI(model="gpt-4", api_key="your-key")

# 绑定工具
llm_with_tools = llm.bind_tools([search_knowledge_base, infer_medical_image])

# 调用示例
if __name__ == "__main__":
    # LLM会自动判断是否需要调用工具
    response = llm_with_tools.invoke("新月体性肾小球肾炎的病理特征是什么？")

    # 如果需要调用工具，LLM会返回tool_calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"调用工具: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")
