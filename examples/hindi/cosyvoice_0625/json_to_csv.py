import json
import csv

json_file = "data_yoyo_sft/test.json"
csv_file = "data_yoyo_sft/test.csv"

# 读取 JSON 文件
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 按 key 排序
sorted_items = sorted(data.items(), key=lambda x: x[0])

# 写入 CSV 文件
with open(csv_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text"])  # 写表头（可选）

    for key, value_list in sorted_items:
        for value in value_list:
            writer.writerow([key, value])

