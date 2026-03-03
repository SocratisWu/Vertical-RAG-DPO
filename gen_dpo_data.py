import json
import random
import re

# --- 配置路径 ---
txt_path = r'E:\LLM_Project\data\full_knowledge.txt'
output_path = r'E:\LLM_Project\dpo_data.jsonl'

def generate_dpo_samples(n=200):
    print(f"开始从 {txt_path} 提取数据并生成 DPO 样本...")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        # 只读取前 10w 行以匹配你的索引规模，并过滤空行
        lines = [line.strip() for i, line in enumerate(f) if line.strip() and i < 100000]

    if len(lines) < n:
        n = len(lines)
    
    samples = random.sample(lines, n)
    dpo_list = []

    for line in samples:
        # 1. 提取核心名称 (假设格式: [场景ID] 类别：名称。内容...)
        match = re.search(r'：(.*?)[。]', line)
        item_name = match.group(1) if match else "该条目"
        
        # 2. 构造 Prompt
        prompt = f"请根据提供的资料，详细介绍一下“{item_name}”。\n资料内容：{line}"
        
        # 3. 构造 Chosen (正面回答：准确、严谨、复述核心事实)
        chosen = f"根据资料显示，{item_name}的相关信息如下：{line.split('。', 1)[-1] if '。' in line else line}。信息完全基于原始记录，确保准确。"
        
        # 4. 构造 Rejected (反面回答：制造常见的 RAG 幻觉)
        # 策略：针对不同类型进行关键词篡改
        rejected = chosen
        
        if "酒店" in line or "景区" in line:
            # 幻觉点：地理位置偏移（比如把南京换成镇江，或者改写星级）
            rejected = rejected.replace("南京", "镇江").replace("5星", "3星").replace("玄武区", "润州区")
        elif "火车" in line or "飞机" in line:
            # 幻觉点：班次/时间篡改（G字头改D字头，下午改凌晨）
            rejected = re.sub(r'[A-Z]\d+', "X999", rejected)
            rejected = rejected.replace(":00", ":59").replace("14:", "03:")
        elif "餐馆" in line or "美食" in line:
            # 幻觉点：评分/评价反转
            rejected = rejected.replace("推荐", "不推荐").replace("好评", "差评").replace("特色", "普通")
        else:
            # 通用策略：否定事实
            rejected = f"资料中没有提到关于{item_name}的具体细节，建议您核实。"

        # 确保 rejected 确实被修改了，如果没有变化，强行添加一个错误描述
        if rejected == chosen:
            rejected = "资料显示该地点目前已关闭，暂不提供服务。"

        dpo_list.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    # 写入 jsonl 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dpo_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ 成功生成 {len(dpo_list)} 条 DPO 数据至: {output_path}")

if __name__ == "__main__":
    generate_dpo_samples(500) # 建议先生成 200 条