"""
真正的ReAct分析脚本
使用qwen3-max（纯语言模型）通过工具调用来分析肾小球图像
- 工具1: infer_medical_image - 图像推理
- 工具2: search_knowledge_base - RAG验证
"""
import os
import json
import requests
from dotenv import load_dotenv
import dashscope
from dashscope import Generation

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = "http://localhost:8001"

dashscope.api_key = DASHSCOPE_API_KEY

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "infer_medical_image",
            "description": "对肾小球病理图片进行诊断。输入一张本地肾小球图片路径，模型会输出诊断结果报告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片的本地相对路径，例如: ./1.png"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在医学知识库中进行RAG检索。输入查询文本，返回相关的医学知识内容。适用于询问肾小球疾病相关的医学问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "查询文本，用英文或中文描述要查询的医学问题"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果的数量，默认10条",
                        "default": 10
                    }
                },
                "required": ["query_text"]
            }
        }
    }
]

# 系统提示
system_prompt = """你是一个专业的肾小球病理分析助手。

你可以使用以下工具：
1. infer_medical_image - 对肾小球病理图片进行诊断
2. search_knowledge_base - 在知识库中检索相关医学知识

请按照ReAct格式思考：
- Thought: 你现在想做什么
- Action: 要调用的工具名
- Action Input: 工具输入参数
- Observation: 工具返回结果

分析流程：
1. 首先调用 infer_medical_image 工具分析图片
2. 根据诊断结果，调用 search_knowledge_base 验证诊断是否正确
3. 最终给出综合结论

注意：你无法直接看到图片，必须调用工具来分析。图片路径是 ./1.png"""

def call_llm(messages):
    """调用qwen3-max"""
    response = Generation.call(
        model='qwen-max',
        messages=messages,
        tools=tools,
        temperature=0.7
    )
    return response

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """执行工具"""
    if tool_name == "infer_medical_image":
        resp = requests.post(
            f"{API_BASE}/infer-medical-image",
            json=tool_input
        )
        return json.dumps(resp.json(), ensure_ascii=False, indent=2)

    elif tool_name == "search_knowledge_base":
        resp = requests.post(
            f"{API_BASE}/search-knowledge-base",
            json=tool_input
        )
        return json.dumps(resp.json(), ensure_ascii=False, indent=2)

    return f"Unknown tool: {tool_name}"

def react_analyze(image_path: str = "./1.png"):
    """ReAct分析流程"""
    print("=" * 60)
    print("ReAct Analysis - Kidney Glomerulus Image")
    print("Image: ./1.png")
    print("=" * 60)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请分析这张肾小球病理图片：{image_path}"}
    ]

    max_steps = 10
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n--- Step {step} ---")

        # 调用LLM
        response = call_llm(messages)

        if response.status_code != 200:
            print(f"LLM Error: {response.message}")
            break

        # 检查是否有工具调用
        try:
            choices = response.output.choices
            if choices:
                msg = choices[0].message

                # 检查是否有工具调用
                try:
                    tool_calls = msg.tool_calls
                except:
                    tool_calls = None

                # 有工具调用
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call['function']['name']
                        tool_args = json.loads(tool_call['function']['arguments'])

                        print(f"Thought: {msg.content}")
                        print(f"Action: {tool_name}")
                        print(f"Action Input: {tool_args}")

                        # 执行工具
                        observation = execute_tool(tool_name, tool_args)
                        print(f"Observation: {observation[:200]}...")

                        # 添加assistant消息和tool结果
                        messages.append({
                            "role": "assistant",
                            "content": msg.content,
                            "tool_calls": [tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": observation
                        })
                else:
                    # 没有工具调用，返回最终结果（可能是思考过程或最终答案）
                    print(f"Thought: {msg.content}")
                    print(f"Final Answer: {msg.content}")
                    print("\n" + "=" * 60)
                    print("Analysis Complete!")
                    print("=" * 60)
                    break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    react_analyze("./1.png")
