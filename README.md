# Vertical-RAG-DPO: 工业级垂直领域抗幻觉问答系统

本项目实现了一套在 **有限算力（单卡 6GB 显存）** 环境下，针对垂直领域（如旅游、酒店）知识库的高效检索与对齐系统。通过 **混合检索 (Hybrid Search)** 与 **直接偏好对齐 (DPO)** 技术，显著解决了小模型在处理海量结构化数据时的幻觉问题。

## 🌟 核心亮点

* **极致资源优化**：基于 **4-bit 量化 (BitsAndBytes)** 技术，将 Qwen2.5-1.5B 模型推理显存压缩至 **2GB 以内**，实现低成本端侧部署。
* **百万级混合检索**：集成 **BGE-M3 语义向量检索** 与 **BM25 关键词检索**，有效应对长尾地理名称与精确路号的召回压力。
* **DPO 抗幻觉微调**：通过自动化构造 Chosen/Rejected 偏好对，应用 **QLoRA** 完成直接偏好对齐，强化模型对 646 万条底层知识的忠诚度。
* **全链路工程化**：包含从 ETL 数据清洗、索引构建到多维单元测试的完整开发流水线。

## 🛠️ 技术栈

* **LLM**: Qwen2.5-1.5B-Instruct
* **Embedding**: BGE-M3
* **微调框架**: TRL, PEFT, Hugging Face
* **检索/索引**: FAISS, Rank-BM25, Jieba
* **性能监控**: 流式输出 (Streamer), TTFT/TPS 实时统计

## 📁 项目结构与脚本说明

```text
├── data/               # 存放原始文本与 FAISS 向量索引
├── qwen_dpo_lora_weights/ # DPO 微调产出的 LoRA 适配器权重
│
├── ingest_data.py      # 【数据工程】异构数据清洗与语义化模板处理
├── build_index.py      # 【索引构建】基于 BGE-M3 的百万级向量索引生成
├── gen_dpo_data.py     # 【数据构造】自动化生成 DPO 偏好对数据集
├── train_dpo.py        # 【模型微调】基于 TRL 的 DPO 训练脚本
├── rag_chat.py         # 【主程序】集成混合检索与 DPO 适配器的对话系统
│
└── tests/              # 【测试链路】
    ├── test_qwen.py    # 基础模型 4-bit 量化后的推理能力验证
    ├── test_rag.py     # RAG 全链路端到端响应测试
    └── test_search.py  # 向量与关键词检索召回率对比测试

监控项,指标,备注
显存占用,~1.85 GB,包含基础模型 + LoRA Adapter
首字延迟 (TTFT),0.4s - 1.1s,受检索速度及 Context 长度影响
吞吐量 (TPS),~35 tokens/s,极速响应，支持流式交互
知识量级,646万条,覆盖 7 类异构业务数据

测试场景,原始模型 (Base),DPO 优化后 (Ours)
诱导性提问,容易顺从错误暗示（如地名张冠李戴）,能够根据资料纠正用户错误，拒绝幻觉
知识盲区,倾向于编造看似合理的虚假事实,直接回答“知识库中暂无相关信息”
指令遵循,回答较为冗长，包含大量通用语,回答简洁专业，严格遵循业务 Prompt

pip install torch transformers peft trl faiss-cpu rank_bm25 jieba bitsandbytes
