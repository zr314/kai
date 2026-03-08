import os
from langchain.embeddings.dashscope import DashScopeEmbeddings

# 正确方式：仅读取环境变量中的 API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
# 额外校验：确保 api_key 不为空且无多余字符
if not api_key or len(api_key) != 32:  # 通义API Key通常是32位
    raise ValueError("DASHSCOPE_API_KEY 格式错误，请检查.env文件")

# 初始化 Embeddings（无多余参数）
embeddings = DashScopeEmbeddings(api_key=api_key)