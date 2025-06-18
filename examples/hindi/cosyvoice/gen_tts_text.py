import json
from tqdm import tqdm

utt2text_path = "data/test/text"
output_json_path = "inference.json"

result = {}

with open(utt2text_path, "r", encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        parts = line.strip().split(' ')
        print("111111111111111: ", parts, len(parts))
        utt_id = parts[0]
        text = " ".join(parts[1:])
        print("2222222222222222: ", text)
        # print(parts)
        # if len(parts) != 2:
        #     continue  # 跳过格式不对的行
        # utt_id, text = parts
        result[utt_id] = [text]  # 用列表包裹文本
print(result, len(result))

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"成功保存到 {output_json_path}，共 {len(result)} 条")
