import os
import glob
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import dashscope
from dashscope import Generation

# 加载环境变量（建议把API-KEY存在.env文件里，避免硬编码）
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # 或直接赋值：dashscope.api_key = "你的API-KEY"


# --------------------------
# 第一步：加载并处理本地TXT文件
# --------------------------
def load_and_split_txt_files(folder_path):
    """
    加载指定文件夹下的所有TXT文件，分割成小文本块（避免单块过长）
    :param folder_path: TXT文件所在文件夹路径
    :return: 分割后的文档列表
    """
    # 1. 遍历文件夹下所有txt文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        raise ValueError(f"文件夹 {folder_path} 下未找到任何TXT文件")

    # 2. 加载每个TXT文件内容
    documents = []
    for file in txt_files:
        try:
            loader = TextLoader(file, encoding="utf-8")  # 注意编码，避免乱码
            docs = loader.load()
            documents.extend(docs)
            print(f"成功加载文件：{file}")
        except Exception as e:
            print(f"加载文件 {file} 失败：{e}")

    # 3. 文本分块（RAG核心：太长的文本无法被LLM有效利用，分块后检索更精准）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个文本块500字符
        chunk_overlap=50,  # 块之间重叠50字符，保证上下文连续
        separators=["\n\n", "\n", "。", "！", "？", "，", " "]  # 按中文分隔符分割
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文本分块完成，共生成 {len(split_docs)} 个文本块")
    return split_docs


# --------------------------
# 第二步：构建并保存向量数据库
# --------------------------
def build_vector_db(split_docs, persist_directory="./chroma_db"):
    """
    将分块后的文本转换为向量，保存到Chroma向量库（本地持久化）
    :param split_docs: 分块后的文档列表
    :param persist_directory: 向量库保存路径
    :return: 构建好的向量数据库
    """
    # 使用通义千问的嵌入模型（免费，适配中文）
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",  # 阿里云通用文本嵌入模型
        api_key=dashscope.api_key
    )

    # 构建向量库并持久化到本地
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 本地保存路径
    )
    vector_db.persist()  # 持久化（Chroma 0.4+版本可省略，但显式调用更稳妥）
    print(f"向量数据库已保存到：{persist_directory}")
    return vector_db


# --------------------------
# 第三步：检索+调用Qwen3.5-Plus API
# --------------------------
def rag_query(vector_db, query, top_k=3):
    """
    1. 从向量库检索与问题最相似的top_k个文本块
    2. 拼接成prompt调用Qwen3.5-Plus API
    :param vector_db: 向量数据库
    :param query: 用户问题
    :param top_k: 检索最相似的k个文本块
    :return: Qwen的回答
    """
    # 1. 检索相似文本
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(query)

    # 2. 拼接检索到的上下文和问题，构建prompt
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
    请基于以下提供的上下文信息回答问题，仅使用上下文里的内容，不要编造信息。如果上下文没有相关信息，请明确说明“无法从提供的资料中找到相关答案”。

    上下文：
    {context}

    问题：{query}
    """

    # 3. 调用Qwen3.5-Plus API
    try:
        response = Generation.call(
            model="qwen-plus",  # qwen3.5-plus的模型标识
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 温度越低，回答越精准（适合RAG）
            result_format="text"
        )
        return response.output.text
    except Exception as e:
        return f"调用API失败：{str(e)}"


# --------------------------
# 主函数：整合所有步骤
# --------------------------
if __name__ == "__main__":
    # 配置参数
    TXT_FOLDER = "./txt_files"  # 你的TXT文件所在文件夹
    VECTOR_DB_PATH = "./chroma_db"  # 向量库保存路径

    # 首次运行：加载TXT→分块→构建向量库（后续只需加载向量库，无需重复构建）
    try:
        # 检查向量库是否已存在，不存在则构建
        if not os.path.exists(VECTOR_DB_PATH):
            split_docs = load_and_split_txt_files(TXT_FOLDER)
            vector_db = build_vector_db(split_docs, VECTOR_DB_PATH)
        else:
            # 已存在向量库，直接加载
            embeddings = DashScopeEmbeddings(model="text-embedding-v1", api_key=dashscope.api_key)
            vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
            print(f"已加载本地向量库：{VECTOR_DB_PATH}")

        # 示例：用户提问
        user_query = "请总结这些TXT文件里的核心内容"
        answer = rag_query(vector_db, user_query)
        print("\n===== 回答结果 =====")
        print(answer)
    except Exception as e:
        print(f"执行失败：{e}")