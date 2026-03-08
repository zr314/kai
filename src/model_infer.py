import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# 固定配置
model_path = "./Qwen3-VL-4B-Instruct"
lora_path = "./qwen3vl-4b-medical-lora-sft"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 固定的医学图像prompt
MEDICAL_PROMPT = "You are provided with a renal pathology image. Based solely on the visual evidence in the image, write a concise renal pathology report-style description using professional medical terminology. Describe findings by anatomical structures. If the findings are insufficient for a definitive diagnosis, use cautious and appropriate language."

# 全局变量缓存模型和处理器
_model = None
_processor = None
_lora_model = None


def load_model():
    """加载模型和处理器"""
    global _model, _processor, _lora_model

    if _model is not None:
        return

    print("加载模型中...", flush=True)

    # 加载基座模型
    _model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # 加载LoRA适配器
    _lora_model = PeftModel.from_pretrained(_model, lora_path)

    # 加载处理器
    _processor = AutoProcessor.from_pretrained(model_path)

    print("模型加载完成", flush=True)


def infer(image_path: str) -> str:
    """
    推理函数

    Args:
        image_path: 图片的本地相对路径

    Returns:
        模型推理结果
    """
    # 确保模型已加载
    load_model()

    # 构造输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": MEDICAL_PROMPT},
            ],
        }
    ]

    # 预处理
    text = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    video_metadatas = None
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)

    inputs = _processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **(video_kwargs or {})
    ).to(device)

    # 生成
    generated_ids = _lora_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True
    )

    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


if __name__ == "__main__":
    # 测试
    result = infer("./1.png")
    print("\n=== 模型回答 ===")
    print(result)
