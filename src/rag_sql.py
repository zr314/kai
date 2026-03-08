# 文本切块
import os
import json
import time
import threading
from openai import OpenAI
from pathlib import Path
from typing import List, Optional
from queue import Queue

# ====================== 核心配置 ======================
# 1. QPS配置（根据阿里云平台配额调整，比如平台配额10，就设10）
MAX_QPS = 10.0
# 2. 并发线程数（建议：QPS≤10时设5-10，QPS>10时设10-20）
MAX_THREADS = 10
# 3. 重试配置
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 1.0  # 重试间隔（秒）
# 4. 任务队列
task_queue = Queue()
# 5. 限流控制
last_call_time = 0.0
qps_lock = threading.Lock()
# 6. 统计
processed_files = 0
total_files = 0
stats_lock = threading.Lock()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def enforce_qps_limit():
    """QPS限流控制（线程安全）"""
    global last_call_time
    with qps_lock:
        min_interval = 1.0 / MAX_QPS
        current_time = time.time()
        time_since_last = current_time - last_call_time
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        last_call_time = time.time()


def read_txt_file(file_path: str) -> str:
    """读取txt文件（兼容多编码）"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""


def validate_json_output(output: str) -> List[str]:
    """验证并解析JSON输出"""
    try:
        json_start = output.find('[')
        json_end = output.rfind(']') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("无有效JSON数组")
        json_str = output[json_start:json_end]
        result = json.loads(json_str)
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise ValueError("JSON格式错误")
        return result
    except Exception as e:
        print(f"JSON解析失败: {e}，原始输出片段: {output[:200]}")
        return []


def call_llm_with_retry(prompt: str) -> Optional[str]:
    """调用LLM并实现重试逻辑"""
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # 执行QPS限流
            enforce_qps_limit()

            completion = client.chat.completions.create(
                model="qwen-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000,
                stream=False,
                timeout=30
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            retry_count += 1
            error_msg = f"LLM调用失败（第{retry_count}次重试）: {str(e)}"
            print(error_msg)

            # 最后一次重试失败时，返回明确的错误信息
            if retry_count == MAX_RETRIES:
                print(f"⚠️  达到最大重试次数（{MAX_RETRIES}次），放弃调用")
                return None

            # 重试前等待（指数退避优化：每次等待时间翻倍）
            sleep_time = RETRY_DELAY * (2 ** (retry_count - 1))
            print(f"等待 {sleep_time:.1f} 秒后重试...")
            time.sleep(sleep_time)

    return None


def process_text_with_llm(raw_text: str) -> List[str]:
    """调用LLM处理文本（包含重试逻辑）"""
    if not raw_text:
        return []

    prompt = """
You are an AI assistant specialized in processing medical texts, particularly related to kidney (renal) diseases and physiology.
Your task is to process the given text according to the following rules:

1. **Translation**: If the text is not in English, translate it into fluent, professional English.
2. **Cleanup**: Correct any garbled characters, typos, or formatting issues to ensure the sentences are clear and grammatically correct.
3. **Relevance Filtering**: If any part of the text is completely unrelated to kidneys (e.g., other organs, general health advice not specific to kidneys, non-medical topics), remove that part entirely.
4. **Splitting**: If a block of text contains multiple distinct facts, ideas, or statements, split it into separate, self-contained sentences. Each output unit should express one complete piece of information about kidneys.
5. **Output Format**: Return the result as a **JSON array of strings**. Each string is one processed sentence/statement. Do not include any additional text, explanations, or markdown formatting.

Need to process the original text:
{}
""".format(raw_text)

    # 调用带重试的LLM接口
    llm_output = call_llm_with_retry(prompt)
    if llm_output is None:
        return []

    # 验证并解析输出
    return validate_json_output(llm_output)


def save_processed_text(processed_list: List[str], original_path: str, output_root: str):
    """保存处理结果"""
    if not processed_list:
        return
    relative_path = os.path.relpath(original_path)
    output_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".json")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_list, f, ensure_ascii=False, indent=2)


def worker(output_root: str):
    """线程工作函数"""
    global processed_files
    while True:
        # 从队列获取任务
        task = task_queue.get()
        if task is None:  # 终止信号
            break

        file_path = task
        try:
            # 读取文本
            raw_content = read_txt_file(file_path)
            if not raw_content:
                print(f"跳过空文件：{file_path}")
                continue

            # 处理文本（已包含重试逻辑）
            processed_list = process_text_with_llm(raw_content)

            # 保存结果
            if processed_list:
                save_processed_text(processed_list, file_path, output_root)
                print(f"完成处理：{file_path}（生成{len(processed_list)}条语句）")
            else:
                print(f"⚠️  文件处理无有效结果：{file_path}（可能是LLM调用失败或解析失败）")

            # 更新统计
            with stats_lock:
                processed_files += 1

        except Exception as e:
            print(f"处理文件{file_path}出错：{str(e)}")
        finally:
            # 标记任务完成
            task_queue.task_done()


def main():
    global total_files, processed_files
    # 配置路径
    INPUT_ROOT = "./rag/txt"
    OUTPUT_ROOT = "./rag/kidney_docs"

    # 1. 扫描所有txt文件
    print("扫描文件中...")
    txt_files = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    total_files = len(txt_files)
    print(f"共扫描到 {total_files} 个txt文件")

    if total_files == 0:
        print("未找到任何txt文件，程序退出")
        return

    # 2. 将文件路径加入任务队列
    for file_path in txt_files:
        task_queue.put(file_path)

    # 3. 启动线程池
    print(f"启动 {MAX_THREADS} 个工作线程...")
    threads = []
    for _ in range(MAX_THREADS):
        t = threading.Thread(target=worker, args=(OUTPUT_ROOT,))
        t.start()
        threads.append(t)

    # 4. 等待所有任务完成
    task_queue.join()

    # 5. 发送终止信号给所有线程
    for _ in range(MAX_THREADS):
        task_queue.put(None)

    # 6. 等待所有线程退出
    for t in threads:
        t.join()

    # 输出统计
    print(f"\n===== 处理完成 ======")
    print(f"总文件数：{total_files}")
    print(f"成功处理：{processed_files}")
    print(f"失败/跳过：{total_files - processed_files}")
    print(f"结果保存路径：{os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    # 检查API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告：未配置DASHSCOPE_API_KEY环境变量！")
        print("Linux/Mac：export DASHSCOPE_API_KEY='你的API Key'")
        print("Windows：set DASHSCOPE_API_KEY=你的API Key")
        exit(1)

    # 启动主程序
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.2f} 秒")