import pandas as pd
import os
import glob

# 基础路径
base_path = r'E:\LLM_Project\data\database_zh'
output_file = r'E:\LLM_Project\data\full_knowledge.txt'

# 定义每个文件类型对应的语义模板，方便 RAG 检索
templates = {
    "attractions": (
        "景点知识条目：{attraction_name}。id为：{attraction_id} "
        "地点：位于{city}（坐标：经度{longitude}，纬度{latitude}）。 "
        "服务详情：评分{rating}分，门票价格为{ticket_price}元。 "
        "开放时间：{opening_time} 至 {closing_time}。 特殊关闭时间:{closing_dates}"
        "游玩建议：建议游玩时长为 {min_visit_time} 到 {max_visit_time} 小时。 "
        "背景介绍：{description},属于{attraction_type}"
    ),
    "flights":(
        "航班知识条目：由{airline}执飞的航班{flight_no}。 "
        "航线：从{origin_city}（{dep_station_name}，代码{dep_station_code}）"
        "飞往{destination_city}（{arr_station_name}，代码{arr_station_code}）。 "
        "时间：出发日期{dep_date}，起飞时间{dep_datetime}，到达时间{arr_datetime}，总飞行耗时{duration}。 "
        "舱位与价格：提供{seat_class}舱位，当前状态为{seat_status}，票价{price}元。 "
        "机型信息：执飞型号为{equip_type}，属于{manufacturer}制造的{equip_size}机型。 "
        "索引参考：航段索引{segment_index}，路由索引{route_index}。"
    ),
    "hotels": (
        "酒店知识条目：{name}。 "
        "档次与品牌：属于{brand}品牌，评级为{hotel_star}星级。 "
        "位置：位于{city}的{address}（坐标：经度{longitude}，纬度{latitude}）。 "
        "设施与评价：综合评分{score}分，参考均价{price}元。 "
        "配套服务：提供{services}。 "
        "时效信息：该酒店于{decoration_time}年进行过装修。"
    ),
    "locations": (
        "地理位置锚点：{poi_name}。 "
        "类型：该地点属于{poi_type}。 "
        "精确坐标：经度{longitude}，纬度{latitude}。"
    ),
    "restaurants": (
        "餐厅知识条目：{restaurant_name}。 "
        "位置与环境：位于{city}（坐标：经度{longitude}, 纬度{latitude}）。 "
        "菜系与消费：主打{cuisine}，人均消费约{price_per_person}元，综合评分{rating}分。 "
        "营业时间：{opening_time} 至 {closing_time}。 "
        "地理关联：该餐厅靠近景点“{nearby_attraction_name}”（景点坐标：{nearby_attraction_coords}）。 "
        "特色标签：{tags}。"
    ),
    "trains": (
        "火车班次条目：{train_type}{train_no}次列车。 "
        "线路：从{origin_city}（{dep_station_name}，站码{dep_station_code}）"
        "开往{destination_city}（{arr_station_name}，站码{arr_station_code}）。 "
        "时间：出发日期{dep_date}，发车时间{dep_datetime}，到达时间{arr_datetime}，全程耗时{duration}。 "
        "坐席与票价：提供{seat_class}，当前状态为{seat_status}，票价{price}元。 "
        "索引信息：路段索引{segment_index}，路由索引{route_index}。"
    ),
    "transportations": (
        "交通接驳信息：从起点坐标({origin})到终点坐标({destination})。 "
        "路程详情：总距离约{distance_meters}米，预计耗时{duration_minutes}分钟。 "
        "花费：预计运输成本为{cost}元。"
    )
}

name_map = {
    "attraction": "attractions",
    "flight": "flights",
    "hotel": "hotels",
    "location": "locations",
    "restaurant": "restaurants",
    "train": "trains",
    "distance": "transportations"
}

all_knowledge = []
print("正在集成 120 个场景的深度知识库...")

# 遍历 id_0 到 id_119
for i in range(120):
    folder_path = os.path.join(base_path, f'id_{i}')
    if not os.path.exists(folder_path):
        continue
        
    # 遍历文件夹下的所有 csv
    for csv_file in glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True):
        raw_file_name = os.path.basename(csv_file).lower()
        
        selected_key = None
        for keyword, t_key in name_map.items():
            if keyword in raw_file_name:
                selected_key = t_key
                break
        
        if selected_key:
            try:
                # 读取 CSV
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                df.columns = [c.strip() for c in df.columns] # 去除列名两端空格
                
                for _, row in df.iterrows():
                    # 1. 将行数据转为字典
                    row_dict = row.to_dict()

                    # 2. 定义容错默认值（针对你刚才报错的那些字段）
                    # 如果 CSV 缺少这些列，或者该格是空的(NaN)，就用默认值补齐
                    defaults = {
                        "tags": "普通",
                        "min_visit_time": "无",
                        "max_visit_time": "无",
                        "services": "常规服务",
                        "description": "暂无介绍",
                        "closing_dates": "无",
                        "attraction_id": "未知",
                        "nearby_attraction_name": "周边景点",
                        "nearby_attraction_coords": "未知"
                    }
                    
                    # 执行补全逻辑
                    for key, val in defaults.items():
                        if key not in row_dict or pd.isna(row_dict[key]):
                            row_dict[key] = val

                    # 3. 填充模板
                    try:
                        # 现在的 row_dict 已经安全了，包含了所有模板需要的字段
                        entry = f"[场景ID {i}] " + templates[selected_key].format(**row_dict)
                        all_knowledge.append(entry)
                    except KeyError as e:
                        # 如果还是缺了某些没定义的关键字段，再记录下来
                        print(f"⚠️ 场景 {i} 的 {raw_file_name} 严重缺失关键字段: {e}")
                        
            except Exception as e:
                print(f"❌ 无法读取文件 {csv_file}: {e}")

# 保存为最终的知识库文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(all_knowledge))

print(f"✅ 集成成功！共处理 {len(all_knowledge)} 条记录。")
print(f"数据已汇总至: {output_file}")