# coding=utf-8
import os
import json
from tqdm import tqdm
import glob



def get_res_list_dict(json_paths):
    acc_list = []      # 统计测评音频的 acc
    real_list = []     # 统计测评音频的 real
    ss_list = []     # 统计测评音频的 spk similarity
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
        ss_list.append(data.get("speaker similarity", 0))
        
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
        
    return acc_list, real_list, ss_list, ref_text_list, user_text_list, unclear_dict, missing_dict, extra_dict


def get_score_from_list(acc_list, real_list, ss_list, ref_text_list, user_text_list, unclear_dict, missing_dict, extra_dict):
    # 计算平均 acc 和 real
    # acc_list = [x for x in acc_list if x != 0]   # 过滤掉为0的值
    # real_list = [x for x in real_list if x != 0]
    # ss_list = [x for x in ss_list if x != 0]
    
    acc_count = len(acc_list)
    real_count = len(real_list)
    ss_count = len(ss_list)
    # assert acc_count == real_count, "acc 和 real 数量不一致"
    avg_acc = sum(acc_list) / acc_count if acc_count > 0 else 0
    avg_real = sum(real_list) / real_count if real_count > 0 else 0
    avg_ss = sum(ss_list) / ss_count if ss_count > 0 else 0
    
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
    
    return avg_acc, avg_real, avg_ss, unclear_wer, miss_extract_wer, wer


def avg_score(save_json_dir, outfile=None):
    res = {}

    eval_spks = os.listdir(save_json_dir)
    # eval_spks = ['SWAGATA', 'Harneet', 'Rajan']
    eval_spks = ['SWAGATA', 'Harneet']
        
    # 遍历每个评测人文件夹
    for eval_spk in eval_spks:
        res[eval_spk] = {}
        spk_dir = os.path.join(save_json_dir, eval_spk)
        if not os.path.isdir(spk_dir):
            continue
        
        
        all_json_files = glob.glob(os.path.join(spk_dir, "*.json"))
        ttype_to_file = {}
        for json_file in all_json_files:
            ttype = json_file.split("_")[-1].replace(".json", "")
            if ttype not in ttype_to_file.keys():
                ttype_to_file[ttype] = []
            ttype_to_file[ttype].append(json_file)
                
        for ttype, json_files in ttype_to_file.items():
            acc_list, real_list, ss_list, ref_text_list, user_text_list, unclear_dict, missing_dict, extra_dict = get_res_list_dict(json_files)
            avg_acc, avg_real, avg_ss, unclear_wer, miss_extract_wer, wer = get_score_from_list(acc_list, real_list, ss_list, ref_text_list, user_text_list, unclear_dict, missing_dict, extra_dict)
            res[eval_spk][ttype] = {"count": len(acc_list), 
                                    "avg_acc": round(avg_acc, 3), 
                                    "avg_real": round(avg_real, 3),
                                    "avg_ss": round(avg_ss, 3),
                                    "wer": round(wer*100, 3),
                                    "unclear_wer": round(unclear_wer*100, 3),
                                    "miss_extract_wer": round(miss_extract_wer*100, 3),
                                    "unclear_dict": unclear_dict,
                                    "missing_dict": missing_dict,
                                    "extra_dict": extra_dict,
                                    "user_text_list": user_text_list,
                                    "len_user_text_list": len(user_text_list)}
        
    
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    
    return res

# save_json_dir = "../output/compare_to_minimax/save_res"
# outfile = "../output/compare_to_minimax/avg_score.txt"

# save_json_dir = "output/yoyo_sft_zz_epoch0_test300_rename/save_res"
# outfile = "output/yoyo_sft_zz_epoch0_test300_rename/save_res/avg_score.txt"

save_json_dir = "../output_test1min/to_eval/save_res"
outfile = "../output_test1min/to_eval/avg_score.txt"

res = avg_score(save_json_dir, outfile)
for spk in res.keys():
    print(f"===== {spk} =====")
    for key in res[spk].keys():
        print(f"{key}: {res[spk][key]}")
    print("\n")


