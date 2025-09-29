# coding=utf-8
import os
import json
from tqdm import tqdm
import glob


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


def get_res_list_dict(json_paths):
    acc_list = []      # 统计测评音频的 acc
    real_list = []     # 统计测评音频的 real
    ref_text_list = [] # 统计参考文本
    user_text_list = []# 统计用户认为的文本对应的data
    unclear_dict = {}  # 统计 unclear 的词汇以及出现次数
    missing_dict = {}  # 统计 missing 的词汇以及出现次数
    extra_dict = {}    # 统计 extra 的词汇以及出现次数
    
    for json_path in tqdm(json_paths):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
            
        if "accuracy" not in data or "realness" not in data:
                continue
            
        acc_list.append(data.get("accuracy", 0))
        real_list.append(data.get("realness", 0))
        
        user_text = data.get("user_text", "")
        if user_text != "":
            user_text_list.append(data)
            
        ref_text = data.get("ref_text", "").strip()
        ref_text_list.append(ref_text)
            
        unclear = data.get("unclear", "").strip()
        if unclear != "":
            if unclear not in unclear_dict.keys():
                unclear_dict[unclear] = 0
            unclear_dict[unclear] += 1
        
        missing = data.get("missing", "").strip()
        if missing != "":
            if missing not in missing_dict.keys():
                missing_dict[missing] = 0
            missing_dict[missing] += 1
        
        extra = data.get("extra", "").strip()
        if extra != "":
            if extra not in extra_dict.keys():
                extra_dict[extra] = 0
            extra_dict[extra] += 1
        
    return acc_list, real_list, ref_text_list, user_text_list, unclear_dict, missing_dict, extra_dict


def get_score_from_list(acc_list, real_list, ref_text_list, user_text_list, unclear_dict, missing_dict, extra_dict):
    # 计算平均 acc 和 real
    acc_count = len(acc_list)
    real_count = len(real_list)
    assert acc_count == real_count, "acc 和 real 数量不一致"
    avg_acc = sum(acc_list) / acc_count if acc_count > 0 else 0
    avg_real = sum(real_list) / real_count if real_count > 0 else 0
    
    # 计算词错误率
    total_words = 0
    for ref_text in ref_text_list:
        total_words += len(ref_text.split(" "))
    total_unclear_words = 0
    for word, count in unclear_dict.items():
        if word != "":
            total_unclear_words += (count * len(word.split(" ")))
    total_missing_words = 0
    for word, count in missing_dict.items():
        if word != "":
            total_missing_words += (count * len(word.split(" ")))
    total_extra_words = 0
    for word, count in extra_dict.items():
        if word != "":
            total_extra_words += (count * len(word.split(" ")))
    
    total_error_words = total_unclear_words + total_missing_words + total_extra_words
    unclear_wer = total_unclear_words / total_words if total_words > 0 else 0
    miss_extract_wer = (total_missing_words + total_extra_words) / total_words if total_words > 0 else 0
    wer = total_error_words / total_words if total_words > 0 else 0
    
    return avg_acc, avg_real, unclear_wer, miss_extract_wer, wer


def avg_score(save_json_dir, outfile=None):
    res = {}

    eval_spks = os.listdir(save_json_dir)
    eval_spks = ['SWAGATA', 'Harneet', 'Rajan']
        
    # 遍历每个评测人文件夹
    for eval_spk in eval_spks:
        res[eval_spk] = {}
        spk_dir = os.path.join(save_json_dir, eval_spk)
        if not os.path.isdir(spk_dir):
            continue
        
        all_json_files = glob.glob(os.path.join(spk_dir, "*.json"))
        gen_json_files = glob.glob(os.path.join(spk_dir, "*_0.json"))
        raw_json_files = [f for f in all_json_files if f not in gen_json_files]
        
        if raw_json_files:
            raw_acc_list, raw_real_list, raw_ref_text_list, raw_user_text_list, raw_unclear_dict, raw_missing_dict, raw_extra_dict = get_res_list_dict(raw_json_files)
            raw_avg_acc, raw_avg_real, raw_unclear_wer, raw_miss_extract_wer, raw_wer = get_score_from_list(raw_acc_list, raw_real_list, raw_ref_text_list, raw_user_text_list, raw_unclear_dict, raw_missing_dict, raw_extra_dict)
            res[eval_spk]["raw"] = {"count": len(raw_acc_list), 
                                    "avg_acc": round(raw_avg_acc, 3), 
                                    "avg_real": round(raw_avg_real, 3),
                                    "wer": round(raw_wer*100, 3),
                                    "unclear_wer": round(raw_unclear_wer*100, 3),
                                    "miss_extract_wer": round(raw_miss_extract_wer*100, 3),
                                    "unclear_dict": raw_unclear_dict,
                                    "missing_dict": raw_missing_dict,
                                    "extra_dict": raw_extra_dict,
                                    "user_text_list": raw_user_text_list}
        
        if gen_json_files:
            gen_acc_list, gen_real_list, gen_ref_text_list, gen_user_text_list, gen_unclear_dict, gen_missing_dict, gen_extra_dict = get_res_list_dict(gen_json_files)
            gen_avg_acc, gen_avg_real, gen_unclear_wer, gen_miss_extract_wer, gen_wer = get_score_from_list(gen_acc_list, gen_real_list, gen_ref_text_list, gen_user_text_list, gen_unclear_dict, gen_missing_dict, gen_extra_dict)
            res[eval_spk]["gen"] = {"count": len(gen_acc_list), 
                                    "avg_acc": round(gen_avg_acc, 3), 
                                    "avg_real": round(gen_avg_real, 3),
                                    "wer": round(gen_wer*100, 3),
                                    "unclear_wer": round(gen_unclear_wer*100, 3),
                                    "miss_extract_wer": round(gen_miss_extract_wer*100, 3),
                                    "unclear_dict": gen_unclear_dict,
                                    "missing_dict": gen_missing_dict,
                                    "extra_dict": gen_extra_dict,
                                    "user_text_list": gen_user_text_list}        
    
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    
    return res

save_json_dir = "../cosyvoice_0625/output/test300_v2/save_res/"
outfile = "../cosyvoice_0625/output/test300_v2/save_res/avg_score.txt"

# save_json_dir = "output/yoyo_sft_zz_epoch0_test300_rename/save_res"
# outfile = "output/yoyo_sft_zz_epoch0_test300_rename/save_res/avg_score.txt"
res = avg_score(save_json_dir, outfile)
for spk in res.keys():
    print(f"===== {spk} =====")
    for key in res[spk].keys():
        print(f"{key}: {res[spk][key]}")
    print("\n")


