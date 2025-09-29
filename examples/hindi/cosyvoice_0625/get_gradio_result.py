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
                
# log_file = "../output_bak.log"
# save_json_dir = "output/test300_v2/save_res/SWAGATA"
# parse_log_to_json(log_file, save_json_dir)


import os
import json

def avg_score(save_json_dir, outfile=None):
    res = {}
    spk_user_text = {}
        
    # 遍历每个评测人文件夹
    for eval_spk in os.listdir(save_json_dir):
        spk_user_text[eval_spk] = []
        spk_dir = os.path.join(save_json_dir, eval_spk)
        if not os.path.isdir(spk_dir):
            continue

        gen_acc_list = []
        gen_real_list = []
        raw_acc_list = []
        raw_real_list = []
        unclear_dict = {}
        missing_dict = {}
        extra_dict = {}
        
        

        # 遍历文件夹下所有 JSON 文件
        for file_name in tqdm(os.listdir(spk_dir)):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(spk_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue  # 读取失败直接跳过

            # 如果缺少 accuracy 或 realness，跳过
            if "accuracy" not in data or "realness" not in data:
                continue

            # 判断是生成音频还是原始音频
            if file_name.endswith("_0.json"):
                gen_acc_list.append(data.get("accuracy", 0))
                gen_real_list.append(data.get("realness", 0))
            else:
                raw_acc_list.append(data.get("accuracy", 0))
                raw_real_list.append(data.get("realness", 0))
            
            user_text = data.get("user_text", "")
            if user_text != "":
                spk_user_text[eval_spk].append(data)
                
            unclear = data.get("unclear", "")
            if unclear not in unclear_dict.keys():
                unclear_dict[unclear] = 0
            unclear_dict[unclear] += 1
            missing = data.get("missing", "")
            if missing not in missing_dict.keys():
                missing_dict[missing] = 0
            missing_dict[missing] += 1
            extra = data.get("extra", "")
            if extra not in extra_dict.keys():
                extra_dict[extra] = 0
            extra_dict[extra] += 1
            

        # 计算平均值
        gen_count = len(gen_acc_list)
        raw_count = len(raw_acc_list)
        gen_avg_acc = sum(gen_acc_list) / gen_count if gen_count > 0 else 0
        gen_avg_real = sum(gen_real_list) / gen_count if gen_count > 0 else 0
        raw_avg_acc = sum(raw_acc_list) / raw_count if raw_count > 0 else 0
        raw_avg_real = sum(raw_real_list) / raw_count if raw_count > 0 else 0

        # 存入结果字典
        res[eval_spk] = {
            "gen": {"count": gen_count, "avg_acc": round(gen_avg_acc, 3), "avg_real": round(gen_avg_real, 3)},
            "raw": {"count": raw_count, "avg_acc": round(raw_avg_acc, 3), "avg_real": round(raw_avg_real, 3)},
            "unclear_dict": unclear_dict,
            "missing_dict": missing_dict,
            "extra_dict": extra_dict
        }
        
    # if outfile is not None:
    #     os.makedirs(os.path.dirname(outfile), exist_ok=True)
    #     with open(outfile, 'w', encoding='utf-8') as f:
    #         for spk, data in res.items():
    #             f.write(f"{spk}: {data}\n\n")
    #         f.write("\n")
    #         for spk, texts in spk_user_text.items():
    #             if len(texts) > 0:
    #                 f.write(spk + "\n")
    #             for text in texts:
    #                 f.write(f"{text}\n")
    #             f.write("\n")
    
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    out_data = {
        "results": res,
        "spk_user_text": spk_user_text
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)
    
    return res


save_json_dir = "output/test300_v2/save_res"
outfile = "output/test300_v2/save_res/avg_score.txt"
res = avg_score(save_json_dir, outfile)
print(res)