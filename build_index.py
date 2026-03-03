import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
import jieba
from rank_bm25 import BM25Okapi

# 1. 配置路径
local_model_path = r'E:\LLM_Project\models\bge-m3'
txt_path = r'E:\LLM_Project\data\full_knowledge.txt'
index_save_path = r'E:\LLM_Project\data\faiss_index.bin'
bm25_save_path = r'E:\LLM_Project\data\bm25_index.pkl'

os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"--- 正在加载 BGE-M3 模型 ---")
model = SentenceTransformer(local_model_path, device='cuda')

# 2. 训练索引参数优化 (针对 646w 数据)
dimension = 1024 
nlist = 1024  # 增加聚类中心，对齐简历“分级索引”
m = 64         
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

# 3. 抽取 10w 条数据训练 (防止聚类中心偏移)
print("正在抽取 10w 样本训练索引...")
train_samples = []
with open(txt_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 40000: break
        line = line.strip()
        if line: train_samples.append(line)

xt = model.encode(train_samples, normalize_embeddings=True)
index.train(np.array(xt).astype('float32'))
print("FAISS 索引训练完成！")

# 4. 全量处理逻辑 (FAISS + BM25 准备)
batch_size = 64 # 提高 Batch 增加吞吐量
count = 0
max_total = 100000 # 开启全量
tokenized_corpus = [] # 用于构建 BM25

print(f"开始全量处理 646w 数据，预计耗时较长，请保持电力充足...")
start_time = time.time()

with open(txt_path, 'r', encoding='utf-8') as f:
    batch_texts = []
    for line in f:
        line = line.strip()
        if not line: continue
        if count >= max_total:
            break

        batch_texts.append(line)
        # 为 BM25 进行分词处理
        tokenized_corpus.append(list(jieba.cut(line))) 

        if len(batch_texts) >= batch_size or (count + len(batch_texts)) >= max_total:
            # A. 向量化并存入 FAISS
            embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            index.add(np.array(embeddings).astype('float32'))
            
            count += len(batch_texts)
            batch_texts = []
            
            # 打印进度
            if count % 1280 == 0:
                elapsed = time.time() - start_time
                speed = count / elapsed
                rem_min = (max_total - count) / speed / 60
                print(f"进度: {count}/{max_total} | 速度: {speed:.1f}条/秒 | 预计剩余: {rem_min:.1f}min")
        if count >= max_total:
            break

    if batch_texts and count < max_total:
        embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        index.add(np.array(embeddings).astype('float32'))
        count += len(batch_texts)
        print(f"处理剩余数据，最终count: {count}/{max_total}")    

# 5. 构建并持久化 BM25 (关键：实现混合检索的基础)
print("正在构建 BM25 索引对象 (646w 分词数据)...")
bm25 = BM25Okapi(tokenized_corpus)
with open(bm25_save_path, 'wb') as f:
    pickle.dump(bm25, f)
print(f"✅ BM25 索引已保存至: {bm25_save_path}")

# 6. 保存 FAISS
print(f"正在保存 FAISS 索引至: {index_save_path}")
faiss.write_index(index, index_save_path)
print(f"✅ 全量混合索引构建任务全部完成！")