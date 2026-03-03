import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 指定下载路径：确保模型下载到 E 盘，不占用 C 盘空间
os.environ['MODELSCOPE_CACHE'] = 'E:/LLM_Project/model_weights'

print("正在准备环境...")

# 2. 下载模型：Qwen2.5-1.5B-Instruct (约 3GB)
# 第一次运行会显示进度条，后续运行会自动跳过下载
model_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct')

print(f"模型已就绪，存放于: {model_dir}")

# 3. 加载分词器和模型
# 对于 1.5B 模型，RTX 3060 的 6G 显存可以直接以 float16 精度加载
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",          # 自动使用 GPU
    torch_dtype=torch.float16   # 使用半精度节省显存
)

# 4. 准备你的第一个问题
prompt = "你好！如果你能看到这条消息，说明我的 RTX 3060 环境配置成功了。请简要自我介绍一下。"
messages = [
    {"role": "system", "content": "你是一个严谨且友好的 AI 开发助手。"},
    {"role": "user", "content": prompt}
]

# 5. 将对话转换为模型能理解的格式
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\n[模型正在思考中...]\n")

# 6. 生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,  # 控制随机性
    top_p=0.8        # 采样范围
)

# 7. 解码并打印答案
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 40)
# 只显示助手的回答部分
if "assistant" in response:
    print(response.split("assistant")[-1].strip())
else:
    print(response)
print("-" * 40)