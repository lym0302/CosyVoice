# coding=utf-8
import os
from modelscope.pipelines import pipeline
import numpy as np
import argparse
import json
import pandas as pd

# 初始化模型
sv_pipeline = pipeline(task="speaker-verification", model="iic/speech_eres2net_sv_zh-cn_16k-common")


def get_spk_similarity(utt_pairs):
    print("111111111111111111111111: ", len(utt_pairs))
    scores = []
    for raw_path, gen_path in utt_pairs:
        try:
            result = sv_pipeline([raw_path, gen_path])
            scores.append(result["score"])
            # print(f"{raw_path} vs {gen_path} similarity: {sim:.4f}")
        except Exception as e:
            print(f"[Warning] Skipped {raw_path} vs {gen_path}: {e}")
    
    mean_score = np.mean(scores) if scores else 0.0
    print(f"\n✅ 平均音色相似度: {mean_score:.4f}")
    return mean_score



def get_utt(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    utts = []
    for key, value_list in data.items():
        if isinstance(value_list, list) and len(value_list) > 0:
            utts.append(key)  
    return utts


def main(args):
    raw_audio_dir = args.raw_audio_dir
    gen_audio_dir = args.gen_audio_dir
    gen_audio_file = args.gen_audio_file
    output_file = args.output_file
    utts = get_utt(args.real_file)
    
    
    suffix_set = set()
    with open(gen_audio_file, "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            wav_path = line.strip()
            key = wav_path.split("/")[-1].replace(".wav", "")
            suffix = key.split("_")[-1]
            suffix_set.add(suffix)
    suffixs = list(suffix_set)
    print("suffixssuffixssuffixs: ", suffixs, len(suffixs))
    
    results = []
    for epoch in suffixs:
        utt_pairs = []
        for utt in utts:
            real_path = f"{raw_audio_dir}/{utt}.wav"
            gen_path = f"{gen_audio_dir}/{utt}_{epoch}.wav"
            utt_pairs.append((real_path, gen_path))
        
        mean_score = get_spk_similarity(utt_pairs)
        
        results.append({
            "epoch": epoch,
            "SS": round(mean_score, 4) if mean_score is not None else "N/A",
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
    parser.add_argument("--real_file", type=str, default="test80.json", help="Path to real.csv")
    parser.add_argument("--raw_audio_dir", type=str, default="output/output_infer/aa", help="Raw audio dir")
    parser.add_argument("--gen_audio_dir", type=str, default="output/output_infer/aa", help="Raw audio dir")
    parser.add_argument("--gen_audio_file", type=str, default="output/output_infer/aa", help="Gen audio file, wav path each line.")
    parser.add_argument("--output_file", type=str, default="output/output_infer_ss.csv", help="Output CSV file to save results")

    args = parser.parse_args()
    
    main(args)
