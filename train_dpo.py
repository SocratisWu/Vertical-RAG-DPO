import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 1. 路径处理（使用您的实际路径）
raw_path = r"E:\LLM_Project\model_weights\models\qwen\Qwen2___5-1___5B-Instruct"
model_path = os.path.abspath(raw_path)

print(f"--- 正在检查模型路径: {model_path} ---")

# 2. 4-bit 量化配置（极致省显存，专为 6G 显存设计）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 3. 加载分词器和模型
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    local_files_only=True, 
    trust_remote_code=True
)
# Qwen 系列必须设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载 4-bit 模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# 4. 准备 LoRA 适配层
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 5. 加载 DPO 数据
dataset = load_dataset("json", data_files="dpo_data.jsonl", split="train")

# 6. 核心配置：使用 DPOConfig（解决 beta 报错的关键）
# 注意：在 TRL 新版本中，beta, max_length 等全部进入 Config
training_args = DPOConfig(
    output_dir="./qwen-dpo-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    logging_steps=1,
    max_steps=100,
    save_steps=50,
    fp16=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    report_to="none",
)

# 7. 启动 DPO 训练
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    # 如果你的 TRL 版本依然报错，请尝试删除下面这一行，
    # 或者确保它能正确识别处理。
    processing_class=tokenizer, 
)

print("🚀 参数校验通过，正式开启 DPO 训练...")
dpo_trainer.train()

# 8. 保存 Adapter 权重
model.save_pretrained("qwen_dpo_lora_weights")
tokenizer.save_pretrained("qwen_dpo_lora_weights")
print("✅ 恭喜！DPO 训练完成，权重已保存至 qwen_dpo_lora_weights")