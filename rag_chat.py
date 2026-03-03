import os
import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel  # <--- 新增：用于加载 Adapter
from sentence_transformers import SentenceTransformer
import time
from transformers import TextIteratorStreamer
from threading import Thread

import jieba
from rank_bm25 import BM25Okapi


# --- 1. 配置路径 ---
bge_path = r'E:\LLM_Project\models\bge-m3'
qwen_path = r'E:\LLM_Project\model_weights\models\qwen\Qwen2___5-1___5B-Instruct' 
dpo_adapter_path = r'./qwen_dpo_lora_weights'  # <--- 新增：DPO 权重路径
index_path = r'E:\LLM_Project\data\faiss_index.bin'
text_data_path = r'E:\LLM_Project\data\full_knowledge.txt'

os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("--- 正在加载 BGE-M3 检索模型... ---")
embed_model = SentenceTransformer(bge_path, device='cuda')

print("--- 正在构建 BM25 索引 (这可能需要一点时间)... ---")
with open(text_data_path, 'r', encoding='utf-8') as f:
    raw_lines = [line.strip() for line in f.readlines() if line.strip()]

# BM25 需要分词后的语料
tokenized_corpus = [list(jieba.cut(doc)) for doc in raw_lines]
bm25 = BM25Okapi(tokenized_corpus)

print("--- 正在加载 FAISS 索引... ---")
index = faiss.read_index(index_path)

print("--- 正在加载 Qwen2.5-1.5B 基础模型 (4-bit)... ---")
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 先加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    qwen_path,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# --- 核心修改：挂载 DPO Adapter ---
print(f"--- 正在挂载 DPO Adapter 权重: {dpo_adapter_path} ---")
qwen_model = PeftModel.from_pretrained(base_model, dpo_adapter_path)
qwen_model.eval() # 切换到推理模式

# --- 2. 检索逻辑 (保持不变) ---
def get_context(query, k=5):
    """混合检索：BM25 + 向量检索"""
    
    # 1. 向量检索 (Semantic Search)
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    vec_distances, vec_indices = index.search(np.array(query_vec).astype('float32'), k)
    vec_results = [raw_lines[i] for i in vec_indices[0] if i < len(raw_lines)]
    
    # 2. BM25 检索 (Keyword Search)
    tokenized_query = list(jieba.cut(query))
    bm25_results = bm25.get_top_n(tokenized_query, raw_lines, n=k)
    
    # 3. 合并去重 (简单的重排逻辑)
    # 优先保留两个检索器都找到的内容，或者交替排列
    combined_results = []
    # 这里使用简单的集合去重，并保持顺序
    seen = set()
    for v, b in zip(vec_results, bm25_results):
        if v not in seen:
            combined_results.append(v)
            seen.add(v)
        if b not in seen:
            combined_results.append(b)
            seen.add(b)
            
    return "\n".join(combined_results[:k]) # 最终返回前 k 条
# --- 3. 对话循环 ---
print("\n✅ RAG + DPO 系统就绪！你可以开始提问了（输入 'quit' 退出）")

while True:
    user_input = input("\n😊 用户: ")
    if user_input.lower() == 'quit':
        break
    
    # A. 检索阶段
    context = get_context(user_input)
    
    # B. 构造 RAG 提示词
    prompt = f"""请根据以下提供的参考资料，简洁、专业地回答用户的问题。
如果参考资料中没有相关信息，请直接回答“抱歉，知识库中暂无相关信息”。

【参考资料】：
{context}

【用户提问】：
{user_input}

【助手回答】："""

    # C. 生成阶段 (流式输出)
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
    streamer = TextIteratorStreamer(qwen_tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=512, 
        do_sample=False, # DPO 后建议用 False，增加确定性
        repetition_penalty=1.1
    )

    start_time = time.time()
    first_token_time = None

    thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
    thread.start()

    print("\n🤖 Qwen(DPO): ", end="", flush=True)
    full_response = ""

    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - start_time
        
        print(new_text, end="", flush=True)
        full_response += new_text

    # 统计性能
    total_time = time.time() - start_time
    # 避免除以零
    gen_time = total_time - ttft if first_token_time else 0.001
    tps = len(qwen_tokenizer.encode(full_response)) / gen_time

    print(f"\n\n[ 性能指标 ] ⏱️ TTFT: {ttft:.2f}s | 🚀 Speed: {tps:.2f} tokens/s")
    # 原代码中此处有重复的 generate 逻辑已删除，直接进入下一轮对话