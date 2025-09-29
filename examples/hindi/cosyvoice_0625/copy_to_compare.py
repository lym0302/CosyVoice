import os
import shutil
import json
from typing import Dict
from tqdm import tqdm
import argparse

def copy_utts_wavs_rename(json_path: str, wav_scp_path: str, target_dir: str):
    # 加载 JSON 文件，获取所需 utt 列表
    with open(json_path, 'r', encoding='utf-8') as f:
        utt2text: Dict[str, list] = json.load(f)
    utts_needed = set(utt2text.keys())

    # 读取 wav.scp 映射关系
    utt2wav = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            utt, path = parts
            utt2wav[utt] = path

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    missing_utts = []
    nogen_utts = []
    for utt in tqdm(utts_needed):
        if utt not in utt2wav:
            print(f"[Warning] {utt} not found in wav.scp, skipped.")
            missing_utts.append(utt)
            continue
            
        src_path = utt2wav[utt]
        if not os.path.isfile(src_path):
            print(f"[Warning] File not found: {src_path}")
            continue

        dst_path = os.path.join(target_dir, f"{utt}_0.wav")
        # dst_path = os.path.join(target_dir, f"{utt}_raw.wav")
        shutil.copy2(src_path, dst_path)

    print(f"\n✅ 拷贝完成：{len(utts_needed) - len(missing_utts) - len(nogen_utts)} 个文件")
    if missing_utts:
        print(f"❗ 缺失 {len(missing_utts)} 个 utt: {missing_utts}")
    if nogen_utts:
        print(f"❗ {len(nogen_utts)} 个 utt 没有生成: {nogen_utts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="data_yoyo_sft/test.json")
    parser.add_argument("--wav_scp_path", type=str, default="data_yoyo_sft/test/wav.scp")
    parser.add_argument("--target_dir", type=str, default="output/output_exp_yoyo_sft_basebbc240_epoch17/aa")
    args = parser.parse_args()

    json_path = args.json_path
    wav_scp_path = args.wav_scp_path
    target_dir = args.target_dir
    copy_utts_wavs_rename(json_path, wav_scp_path, target_dir)