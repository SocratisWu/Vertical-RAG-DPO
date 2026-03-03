import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载模型和索引
model_path = r'E:\LLM_Project\models\bge-m3'
index_path = r'E:\LLM_Project\data\faiss_index.bin'
model = SentenceTransformer(model_path, device='cuda')
index = faiss.read_index(index_path)

# 2. 模拟用户提问
query = "这里输入一个你知识库里肯定有的关键词"
query_vector = model.encode([query], normalize_embeddings=True)

# 3. 检索前 5 条最相关的
top_k = 5
distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)

print(f"最相关的行号索引: {indices[0]}")
print(f"相似度距离: {distances[0]}")

# 在 test_search.py 结尾增加这一段
def show_results(indices):
    with open(r'E:\LLM_Project\data\full_knowledge.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("\n--- 检索到的参考资料 ---")
        for i, idx in enumerate(indices[0]):
            if idx < len(lines):
                print(f"[{i+1}] {lines[idx].strip()}")

show_results(indices)