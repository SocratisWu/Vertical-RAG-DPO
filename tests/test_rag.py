import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 路径与环境设置
os.environ['MODELSCOPE_CACHE'] = 'E:/LLM_Project/model_weights'
# 确保数据文件夹路径正确
DATA_PATH = "E:/LLM_Project/data/knowledge.txt"

print("正在准备模型和 Embedding 环境...")
MODEL_DIR = snapshot_download('qwen/Qwen2.5-1.5B-Instruct')
# 使用一个非常轻量且对中文支持极好的 Embedding 模型
print("正在加载 BGE Embedding 权重（安全格式）...")
embedding_model_path = snapshot_download("BAAI/bge-small-zh-v1.5")

# 2. 加载生成模型 (Qwen 1.5B)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, device_map="auto", torch_dtype=torch.float16
)

# 3. 初始化 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True} # BGE 模型建议开启归一化
)

# 4. 读取并切分知识库
print(f"正在读取文档: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"错误：找不到文件 {DATA_PATH}，请检查路径！")
    exit()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text_content = f.read()

# RAG 的关键：切片。这里设置每个块 200 字，重叠 30 字保证语义连续
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = text_splitter.split_text(text_content)

# 5. 构建向量数据库
print("正在构建向量索引...")
vector_db = FAISS.from_texts(chunks, embeddings)

# 6. RAG 问答逻辑
def ask_rag(query):
    # 检索最相关的 3 条内容
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 构建 Prompt：给模型下达明确指令
    prompt = f"""你是一个专业的运维助手。请仅根据提供的参考信息回答问题。
若参考信息中未提及，请诚实回答“手册中未记录此信息”。

参考信息：
{context}

用户问题：
{query}
"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回答
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.1)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.split("assistant")[-1].strip()

# --- 循环测试环节 ---
print("\n" + "="*30)
print("RAG 系统已就绪！请输入问题（输入 'exit' 退出）")
while True:
    question = input("\n您的提问: ")
    if question.lower() == 'exit':
        break
    
    result = ask_rag(question)
    print(f"\n[AI 回答]: {result}")