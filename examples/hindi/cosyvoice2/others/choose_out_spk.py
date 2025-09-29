# coding=utf-8
import re
import torch
from tqdm import tqdm

import os
import re
import requests
from tqdm import tqdm

def download_audios(txt_path, save_dir="aa"):
    os.makedirs(save_dir, exist_ok=True)

    # 一次性读入文本
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 按 JSON 块分割（每个 {}）
    blocks = re.findall(r'\{.*?\}', text, flags=re.S)

    for block in tqdm(blocks, desc="Downloading"):
        # 提取 user_id
        uid_match = re.search(r'"user_id"\s*:\s*NumberLong\(\s*"?(\d+)"?\s*\)', block)
        if not uid_match:
            continue
        uid = uid_match.group(1)

        # 提取 voice_url
        url_match = re.search(r'"voice_url"\s*:\s*"([^"]+)"', block)
        # 提取 voice_16k_url
        url16_match = re.search(r'"voice_16k_url"\s*:\s*"([^"]+)"', block)

        # 下载 voice_url → 48k
        if url_match:
            url = url_match.group(1)
            save_path = os.path.join(save_dir, f"{uid}_ref_48k.wav")
            download_file(url, save_path)

        # 下载 voice_16k_url → 16k
        if url16_match:
            url16 = url16_match.group(1)
            save_path = os.path.join(save_dir, f"{uid}_ref_16k.wav")
            download_file(url16, save_path)


def download_file(url, save_path):
    """带进度条的下载"""
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(save_path, "wb") as f, tqdm(
        desc=f"Saving {os.path.basename(save_path)}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))



def extract_user_ids_from_txt(txt_path):
    """
    逐行读取 txt 文件，提取所有 user_id (NumberLong 里的数字)，返回字符串列表
    带 tqdm 进度条
    """
    user_ids = []

    # 先数一下文件行数，用于 tqdm total
    with open(txt_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            match = re.search(r'"user_id"\s*:\s*NumberLong\(\s*"?(\d+)"?\s*\)', line)
            if match:
                spk = match.group(1)
                print("match: ", match)
                print("spk: ", spk)
                user_ids.append(spk)  # 作为字符串

    return user_ids


# 使用示例
if __name__ == "__main__":
    # 已有的 user_id 列表
    txt_path = "long_text_922968de-5a51-4173-8b0b-ef35819ff624.txt"
    spk_to_emb = torch.load("../datas/v1_1000h-bbc_v1_240/train/spk2info.pt")
    in_spks = spk_to_emb.keys() 
    all_spks = extract_user_ids_from_txt(txt_path)
    out_spks = [spk for spk in all_spks if spk not in in_spks]

    print("总 user_id 数量:", len(all_spks))
    print("集外 spks:", out_spks, len(out_spks))
    
    download_audios(txt_path, "minimax_ref_audios")
