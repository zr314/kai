# 生成向量
import os
import json
import time
import glob
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import dashscope
from dashscope import TextEmbedding

# ====================== 配置项 ======================
load_dotenv()

# 1. 阿里云百炼配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_QPS = 10  # 平台QPS配额
MAX_WORKERS = 10  # 线程数=QPS
BATCH_SIZE = 100  # 线程批次大小

# 2. 文件路径
JSON_ROOT = "./rag/kidney_docs"  # 原始JSON文件目录
OUTPUT_VECTOR_FILE = "./kidney_vectors.json"  # 生成的向量文件

# ====================== QPS限流（线程安全） ======================
import threading

last_call_time = 0.0
qps_lock = threading.Lock()


def enforce_qps_limit():
    global last_call_time
    with qps_lock:
        min_interval = 1.0 / EMBEDDING_QPS
        current_time = time.time()
        time_since_last = current_time - last_call_time
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        last_call_time = current_time


# ====================== 生成向量并保存 ======================
def get_embedding(text: str, source: str) -> dict:
    """生成单条向量"""
    max_retries = 2
    for retry in range(max_retries):
        try:
            enforce_qps_limit()
            response = TextEmbedding.call(
                model=EMBEDDING_MODEL,
                input=text,
                text_type="document",
                output_format="float"
            )
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                if len(embedding) == 1024:  # text-embedding-v4维度
                    return {
                        "text": text,
                        "embedding": embedding,
                        "source": source,
                        "text_length": len(text)
                    }
        except Exception as e:
            if retry == max_retries - 1:
                print(f"❌ 失败: {text[:30]}... | {e}")
                return None
            time.sleep(0.5 * (retry + 1))
    return None


def load_raw_data() -> list:
    """加载原始文本数据"""
    data_list = []
    json_files = glob.glob(os.path.join(JSON_ROOT, "**/*.json"), recursive=True)
    print(f"📁 找到 {len(json_files)} 个原始文件")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            if isinstance(json_data, list):
                for text in json_data:
                    text = text.strip()
                    if text and len(text) <= 4000:
                        data_list.append((text, file_path))
        except Exception as e:
            print(f"⚠️  读取失败 {file_path}: {e}")

    print(f"📝 共加载 {len(data_list)} 条文本")
    return data_list


def batch_generate_vectors():
    """批量生成向量并保存为JSON"""
    # 1. 加载原始数据
    data_list = load_raw_data()
    if not data_list:
        return

    # 2. 多线程生成向量
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in tqdm(range(0, len(data_list), BATCH_SIZE), desc="🚀 生成向量"):
            batch = data_list[i:i + BATCH_SIZE]
            futures = [executor.submit(get_embedding, text, source) for text, source in batch]

            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)

    # 3. 保存到JSON文件（每行一个JSON对象，Milvus支持的格式）
    print(f"\n💾 保存 {len(all_results)} 条向量到 {OUTPUT_VECTOR_FILE}")
    with open(OUTPUT_VECTOR_FILE, 'w', encoding='utf-8') as f:
        for item in all_results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')  # 每行一个JSON，便于Milvus导入

    print(f"✅ 向量文件生成完成！")


if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        print("❌ 未配置DASHSCOPE_API_KEY！")
        exit(1)
    batch_generate_vectors()