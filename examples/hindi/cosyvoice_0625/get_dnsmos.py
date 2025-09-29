#coding=utf-8

import argparse
import json
import pandas as pd
import csv
import os


def read_csv(csv_file):
    utt_p808mos = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            utt = filename.split("/")[-1].replace(".wav", "")
            p808_mos = float(row["P808_MOS"])
            utt_p808mos[utt] = p808_mos
    return utt_p808mos


def get_utt(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    utts = []
    for key, value_list in data.items():
        if isinstance(value_list, list) and len(value_list) > 0:
            utts.append(key)  
    return utts


def main(args):
    utts = get_utt(args.json_file)
    output_file = args.output_file
    utt_p808mos = read_csv(args.csv_file)
    
    suffix_set = set()
    for key in utt_p808mos.keys():
        suffix = key.split("_")[-1]
        suffix_set.add(suffix)
    suffixs = list(suffix_set)
    print("suffixssuffixssuffixs: ", suffixs, len(suffixs))
    
    results = []
    for epoch in suffixs:
        scores = []
        for utt in utts:
            utt = f"{utt}_{epoch}"
            scores.append(utt_p808mos[utt])
        mean_score = sum(scores)/len(scores)
        
        results.append({
            "epoch": epoch,
            "DNSMOS": round(mean_score, 4) if mean_score is not None else "N/A",
        })

    # df = pd.DataFrame(results)
    # df.to_csv(args.output_file, index=False)
    df = pd.DataFrame(results) 
    df['epoch'] = df['epoch'].astype(int)
    df = df.sort_values(by='epoch')  # 根据 epoch 数值进行从小到大排序
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False)  # 如果文件存在就追加，不写入表头
    else:
        df.to_csv(output_file, mode='w', index=False, header=True)   # 如果文件不存在就新建，写入表头
    
    print(df)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WER on kept hi-IN predictions by confidence threshold.")
    parser.add_argument("--json_file", type=str, default="test80.json", help="Path to real.csv")
    parser.add_argument("--csv_file", type=str, default="output/output_infer_dnsmos.csv", help="Path to pred.csv")
    parser.add_argument("--output_file", type=str, default="output/output_infer_dnsmos_avg.csv", help="Output CSV file to save results")

    args = parser.parse_args()
    
    main(args)
