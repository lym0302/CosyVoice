#codind=utf-8

import os
import json
import glob
from tqdm import tqdm

asr_result_dir = 'tttest_raw_asr'
output_file = 'raw_asr.list'

asr_txt_paths = glob.glob(f"{asr_result_dir}/*.txt")

with open(output_file, 'w', encoding='utf-8') as fout:
    for asr_txt_path in tqdm(asr_txt_paths):
        
        utt = asr_txt_path.split("/")[-1].replace(".txt", "")
        spk = utt.split("_")[0]
        
        if not os.path.exists(asr_txt_path):
            print(f"[Warning] Missing ASR file: {asr_txt_path}")
            continue

        try:
            if os.path.getsize(asr_txt_path) == 0:
                print(f"[Warning] Empty ASR file: {asr_txt_path}")
                continue
            
            with open(asr_txt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取 nBest 列表
            n_best_list = data.get("recognizedPhrases", [])[0].get("nBest", [])
            confidence = n_best_list[0].get("confidence", 0.0)
            text = n_best_list[0].get("display", "")

            fout.write(f"{utt}\t{spk}\t{text}\t{confidence:.4f}\n")

        except json.JSONDecodeError as je:
            print(f"[Error] JSON parsing failed: {asr_txt_path} - {je}")
        except Exception as e:
            print(f"[Error] Failed to process {asr_txt_path}: {e}")
