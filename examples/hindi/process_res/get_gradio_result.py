# coding=utf-8
import os
import json
from tqdm import tqdm


def jsonl_to_json(jsonl_file, save_json_dir):
    os.makedirs(save_json_dir, exist_ok=True)
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            data = json.loads(line)  # 每行转为 dict
            wav_path = data.get("wav_path")
            if wav_path is None:
                continue  # 没有 wav_path 就跳过
            
            utt = wav_path.split("/")[-1].replace(".wav", "")
            json_path = os.path.join(save_json_dir, f"{utt}.json")
            
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(data, jf, ensure_ascii=False, indent=2)

# jsonl_file = "output/test300_v2/results/res_SWAGATA_6u01tr_301_400.jsonl"
# save_json_dir = "output/test300_v2/save_res/SWAGATA"
# jsonl_to_json(jsonl_file, save_json_dir)


import ast

def parse_log_to_json(log_file, save_json_dir):
    os.makedirs(save_json_dir, exist_ok=True)

    with open(log_file, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            if "SWAGATA_crgazz" not in line:
                continue

            try:
                # 提取 wav_path
                wav_key = "wav_path: "
                data_key = "data: "

                wav_start = line.index(wav_key) + len(wav_key)
                wav_end = line.index(" - data: ")
                wav_path = line[wav_start:wav_end].strip()

                # 提取 data dict
                data_start = line.index(data_key) + len(data_key)
                data_str = line[data_start:].strip()
                data_dict = ast.literal_eval(data_str)  # 安全转换成 dict

                # 构造结果
                record = {"wav_path": wav_path}
                record.update(data_dict)

                # 生成文件名
                utt = wav_path.split("/")[-1].replace(".wav", "")
                json_path = os.path.join(save_json_dir, f"{utt}.json")

                # 保存 json
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(record, jf, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"解析失败: {line[:100]}... 错误: {e}")
                
log_file = "../output_bak.log"
save_json_dir = "output/test300_v2/save_res/SWAGATA"
parse_log_to_json(log_file, save_json_dir)