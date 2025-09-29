# coding=utf-8

import hashlib
import csv
import os
from collections import defaultdict
from pydub.utils import mediainfo
import argparse
from tqdm import tqdm

def client_id_to_spk_id(client_id: str, length: int = 8, prefix: str = "spk") -> str:
    """
    将 client_id 映射为长度为 length 的短 spk_id。
    """
    hash_str = hashlib.md5(client_id.encode('utf-8')).hexdigest()
    return f"{prefix}_{hash_str[:length]}"

def extract_duration(wav_path: str) -> float:
    """
    使用 pydub 计算 wav 音频时长（单位：秒）
    """
    try:
        info = mediainfo(wav_path)
        return float(info["duration"])
    except Exception as e:
        print(f"⚠️ 无法获取音频时长: {wav_path}, 错误: {e}")
        return 0.0

def process_data_tsv(tsv_path: str, 
                     audio_dir: str,
                     clientid2spkid_path: str = "clientid2spkid.tsv",
                     data_list_path: str = "data.list"):
    clientid2spk = {}
    data_list = []

    with open(tsv_path, encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in tqdm(reader):
            client_id = row["client_id"].strip()
            path = row["path"].strip()
            sentence = row["sentence"].strip()

            if not client_id or not path or not sentence:
                continue

            # 映射 spk_id
            if client_id not in clientid2spk:
                clientid2spk[client_id] = client_id_to_spk_id(client_id)

            spk_id = clientid2spk[client_id]
            wav_path = os.path.join(audio_dir, path).replace(".mp3", ".wav")
            duration = extract_duration(wav_path)

            line = f"{wav_path}\t{spk_id}\t{sentence}\t{duration:.3f}\t1.0\n"
            data_list.append(line)

    # 保存 clientid2spkid.tsv
    with open(clientid2spkid_path, "w", encoding="utf-8") as f:
        for cid, sid in clientid2spk.items():
            f.write(f"{cid}\t{sid}\n")

    # 保存 data.list
    with open(data_list_path, "w", encoding="utf-8") as f:
        f.writelines(data_list)

    print(f"✅ 映射完成，共生成 {len(data_list)} 条数据。")
    print(f"📝 clientid2spkid 保存到: {clientid2spkid_path}")
    print(f"📝 data.list 保存到: {data_list_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 data.tsv 生成 spk_id 映射和 data.list 文件")
    parser.add_argument("-i", "--tsv_path", required=True, help="输入的 data.tsv 路径")
    parser.add_argument("-a", "--audio_dir", required=True, help="包含 .wav 文件的目录")
    parser.add_argument("-c", "--clientid2spkid_path", default="clientid2spkid.tsv", help="输出的 clientid2spkid 映射文件路径")
    parser.add_argument("-o", "--data_list_path", default="data.list", help="输出的 data.list 路径")

    args = parser.parse_args()
    process_data_tsv(args.tsv_path, args.audio_dir, args.clientid2spkid_path, args.data_list_path)
