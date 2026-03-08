import os
import json
from dotenv import load_dotenv
from pymilvus import connections, Collection
import dashscope
from dashscope import TextEmbedding

# ====================== 配置项 ======================
load_dotenv()

# 1. Milvus配置（和导入时一致）
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "kidney_rag_collection"
DIMENSION = 1024  # text-embedding-v4的维度

# 2. 阿里云百炼配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v4"

# 3. 检索配置
TOP_K = 10  # 返回10条相似结果
METRIC_TYPE = "COSINE"  # 余弦相似度（和入库时一致）

# ====================== 初始化 ======================
# 初始化百炼SDK
dashscope.api_key = DASHSCOPE_API_KEY

def init_milvus():
    """连接Milvus并加载集合（适配pymilvus 2.4.4）"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        collection = Collection(COLLECTION_NAME)
        collection.load()
        num_entities = collection.num_entities
        print(f"✅ 成功连接Milvus集合：{COLLECTION_NAME}")
        print(f"   - 集合总数据量：{num_entities} 条")
        return collection
    except Exception as e:
        print(f"❌ Milvus初始化失败：{e}")
        return None

# ====================== 生成查询向量 ======================
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

# ====================== 检索并打印结果 ======================
def search_crescent_lesion(collection: Collection):
    """查询“新月体病变特征”的10条相关数据（最终适配版）"""
    if collection is None:
        return

    # 策略好新月体性肾小球肾炎病理结构组成
    # 1. 定义查询文本（英文）
    query_text = "crescentic glomerulonephritis pathological structure composition"
    print(f"\n🔍 检索查询：{query_text}")
    print("=" * 100)

    # 2. 生成查询向量
    query_embedding = get_query_embedding(query_text)
    if query_embedding is None:
        print("❌ 查询向量生成失败")
        return

    # 3. 构建检索参数
    search_params = {
        "metric_type": METRIC_TYPE,
        "params": {"nprobe": 50}
    }

    # 4. 执行检索
    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["text", "source", "text_length"]
        )

        # 5. 解析结果（✅ 核心修复：用try-except替代has()）
        if results and len(results[0]) > 0:
            print(f"\n📌 共检索到 {len(results[0])} 条相关结果（按相似度排序）：")
            print("=" * 100)

            for i, hit in enumerate(results[0], 1):
                # 2.4.4版本：直接get，捕获字段不存在异常
                try:
                    text = hit.entity.get("text")
                except:
                    text = "未知"
                try:
                    source = hit.entity.get("source")
                except:
                    source = "未知"
                try:
                    text_length = hit.entity.get("text_length")
                except:
                    text_length = "未知"

                print(f"\n【第{i}条】相似度：{hit.score:.4f}")
                print(f"原文本：{text}")
                print(f"来源文件：{source}")
                print(f"文本长度：{text_length} 字符")
                print("-" * 80)
        else:
            print("❌ 未检索到相关数据")

    except Exception as e:
        print(f"❌ 检索出错：{e}")

# ====================== 主函数 ======================
if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        print("❌ 未配置DASHSCOPE_API_KEY环境变量！")
        exit(1)

    collection = init_milvus()
    search_crescent_lesion(collection)