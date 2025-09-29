# coding=utf-8

import os
import pandas as pd
import jiwer
import re
import argparse
import json

# ---------- 清洗器 ----------
def remove_hindi_punctuation(text: str) -> str:
    return re.sub(r"[।॥!?.,\"'“”‘’]", "", text)

# ---------- 加载数据 ----------
# def load_hiIN_transcriptions(path):
#     data = {}
#     with open(path, encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if row["Language"].strip() == "hi-IN":
#                 filename = row["Filename"].strip()
#                 text = row["Transcription"].strip()
#                 conf = float(row["Confidence"]) if row["Confidence"].strip() else 0.0
#                 data[filename] = (text, conf)
#     return data

def get_real(real_json):
    with open(real_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    real_dict = {}
    # utts = []
    for key, value_list in data.items():
        if isinstance(value_list, list) and len(value_list) > 0:
            real_dict[key] = (value_list[0], 1.0)
            # utts.append(key)
            
    return real_dict


def get_pred(pred_file):
    pred_dict = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            wav_path, spk, text, dur, asr_conf = line.strip().split("\t")
            utt = wav_path.split("/")[-1].replace(".wav", "")
            pred_dict[utt] = (text, float(asr_conf))  # 注意：asr_conf 转为 float 比较常见
    return pred_dict



# ---------- 分析 WER ----------
def analyze_kept_wer(real_path, pred_path, thresholds, output_file):
    real_data = get_real(real_path)
    print("1111111111111111: ", len(real_data))
    pred_data = get_pred(pred_path)
    print("2222222222222222222: ", len(pred_data))
    
    suffix_set = set()
    for key in pred_data.keys():
        suffix = key.split("_")[-1]
        suffix_set.add(suffix)
    suffixs = list(suffix_set)
    print("suffixssuffixssuffixs: ", suffixs, len(suffixs))

    results = []
    
    # suffixs = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10", "_11", "_12"]
    for epoch in suffixs:
        kept_refs = []
        kept_hyps = []
        kept_refs_havepunc = []
        kept_hyps_havepunc = []
        kept_count = 0

        for utt, (ref, conf) in real_data.items():
            gen_utt = f"{utt}_{epoch}"
            if gen_utt in pred_data:
                hyp, _ = pred_data[gen_utt]
                kept_refs.append(remove_hindi_punctuation(ref))
                kept_hyps.append(remove_hindi_punctuation(hyp))
                kept_refs_havepunc.append(ref)
                kept_hyps_havepunc.append(hyp)
                kept_count += 1

        if kept_refs:
            wer_value = jiwer.wer(
                kept_refs,
                kept_hyps
            )
            wer_value_havapunc = jiwer.wer(
                kept_refs_havepunc,
                kept_hyps_havepunc
            )
        else:
            wer_value = None
            wer_value_havapunc = None

        results.append({
            "epoch": epoch,
            "Kept_Count": kept_count,
            "WER_Kept have punc": round(wer_value_havapunc, 4) if wer_value_havapunc is not None else "N/A",
            "WER_Kept": round(wer_value, 4) if wer_value is not None else "N/A"
        })
    
    df = pd.DataFrame(results) 
    df['epoch'] = df['epoch'].astype(int)
    df = df.sort_values(by='epoch')  # 根据 epoch 数值进行从小到大排序
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False)  # 如果文件存在就追加，不写入表头
    else:
        df.to_csv(output_file, mode='w', index=False, header=True)   # 如果文件不存在就新建，写入表头

    
    # df.to_csv(output_file, index=False)
    print(df)

# ---------- 主函数入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WER on kept hi-IN predictions by confidence threshold.")
    parser.add_argument("--real_file", type=str, default="test80.json", help="Path to real.csv")
    parser.add_argument("--pred_file", type=str, default="output/output_infer.list", help="Path to pred.csv")
    parser.add_argument("--output_file", type=str, default="output/output_infer_wer.csv", help="Output CSV file to save results")

    args = parser.parse_args()

    # thresholds = [round(i * 0.1, 1) for i in range(10)]
    thresholds = [0.0]
    analyze_kept_wer(args.real_file, args.pred_file, thresholds, args.output_file)
