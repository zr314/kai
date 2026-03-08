# 导入数据库
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# ====================== 配置项 ======================
load_dotenv()

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "kidney_rag_collection"
DIMENSION = 1024

# 文件路径
VECTOR_FILE_PATH = "./kidney_vectors.json"  # 生成的向量文件
BATCH_SIZE = 200  # 批量插入大小（增大批次提高速度）


# ====================== 导入Milvus（通用方案） ======================
def init_milvus_collection():
    """创建Milvus集合"""
    # 连接Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # 删除已有集合
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="text_length", dtype=DataType.INT64)
    ]

    # 创建集合
    schema = CollectionSchema(fields, description="Kidney RAG Vector DB")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✅ Milvus集合 {COLLECTION_NAME} 创建成功")
    return collection


def load_vector_data():
    """加载JSON向量文件（每行一个JSON对象）"""
    if not os.path.exists(VECTOR_FILE_PATH):
        print(f"❌ 向量文件不存在：{VECTOR_FILE_PATH}")
        return []

    data_list = []
    print(f"\n📥 开始加载向量文件：{VECTOR_FILE_PATH}")
    with open(VECTOR_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载数据"):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # 校验必要字段
                    if all(k in item for k in ["text", "embedding", "source", "text_length"]):
                        data_list.append(item)
                except Exception as e:
                    print(f"\n⚠️  跳过错误行：{line[:50]}... | 错误：{e}")

    print(f"✅ 成功加载 {len(data_list)} 条有效向量数据")
    return data_list


def batch_insert_data(collection: Collection, data_list: list):
    """批量插入数据到Milvus"""
    if not data_list:
        return

    total_inserted = 0
    # 分批次插入
    for i in tqdm(range(0, len(data_list), BATCH_SIZE), desc="📤 导入Milvus"):
        batch = data_list[i:i + BATCH_SIZE]

        # 整理批量数据
        texts = []
        embeddings = []
        sources = []
        text_lengths = []

        for item in batch:
            texts.append(item["text"])
            embeddings.append(item["embedding"])
            sources.append(item["source"])
            text_lengths.append(item["text_length"])

        # 执行插入
        try:
            collection.insert([texts, embeddings, sources, text_lengths])
            total_inserted += len(texts)

            # 每10批刷一次盘（减少IO开销）
            if i % (BATCH_SIZE * 10) == 0 and i > 0:
                collection.flush()
        except Exception as e:
            print(f"\n❌ 批量{i}-{i + BATCH_SIZE}插入失败：{e}")
            continue

    # 最后统一刷盘
    collection.flush()
    print(f"\n✅ 批量插入完成！成功插入 {total_inserted} 条数据")


def create_index(collection: Collection):
    """创建向量索引（提高检索速度）"""
    print("\n🔧 开始创建向量索引...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()  # 加载集合到内存
    print(f"✅ 索引创建完成！集合总数据量：{collection.num_entities} 条")


if __name__ == "__main__":
    # 1. 初始化集合
    collection = init_milvus_collection()

    # 2. 加载向量数据
    data_list = load_vector_data()

    # 3. 批量插入
    batch_insert_data(collection, data_list)

    # 4. 创建索引
    create_index(collection)

    # 5. 最终统计
    print(f"\n📈 导入完成总结：")
    print(f"   - Milvus集合名称：{COLLECTION_NAME}")
    print(f"   - 总数据量：{collection.num_entities} 条")
    print(f"   - 向量维度：{DIMENSION}")
    print(f"   - Milvus控制台：http://localhost:8000")