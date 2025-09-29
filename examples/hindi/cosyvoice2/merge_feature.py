import os
import torch
from collections import defaultdict
import argparse
import subprocess

def load_spk2utt(path):
    spk2utt = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            spk = parts[0]
            utts = parts[1:]
            spk2utt[spk].extend(utts)
    return spk2utt

def save_spk2utt(spk2utt, path):
    with open(path, "w") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")

def merge_pt_dicts(path_a, path_b, output_path):
    dict_a = torch.load(path_a)
    dict_b = torch.load(path_b)
    dict_c = dict_a.copy()
    dict_c.update(dict_b)
    torch.save(dict_c, output_path)

def compute_spk2utt_count(spk2utt):
    return {spk: len(utts) for spk, utts in spk2utt.items()}

def merge_spk2embedding(a_path, b_path, a_spk2utt, b_spk2utt, output_path):
    emb_a = torch.load(a_path)
    emb_b = torch.load(b_path)
    count_a = compute_spk2utt_count(a_spk2utt)
    count_b = compute_spk2utt_count(b_spk2utt)

    merged = {}

    for spk in set(list(emb_a.keys()) + list(emb_b.keys())):
        in_a = spk in emb_a
        in_b = spk in emb_b
        if in_a and in_b:
            wa = count_a.get(spk, 0)
            wb = count_b.get(spk, 0)
            if wa + wb == 0:
                merged[spk] = emb_a[spk]
            else:
                merged[spk] = (emb_a[spk] * wa + emb_b[spk] * wb) / (wa + wb)
        elif in_a:
            merged[spk] = emb_a[spk]
        elif in_b:
            merged[spk] = emb_b[spk]
    torch.save(merged, output_path)
    


def cat_merge_files(file_name, a_dir, b_dir, c_dir):
    a_file = os.path.join(a_dir, file_name)
    b_file = os.path.join(b_dir, file_name)
    c_file = os.path.join(c_dir, file_name)

    # 使用 shell 的 cat 命令拼接
    cmd = f"cat {a_file} {b_file} > {c_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ 用 cat 拼接完成：{file_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ cat 拼接失败：{file_name}")
        print(e)
        

def check_spk(a_spk2utt, b_spk2utt):
    common_spks = list(set(a_spk2utt.keys()) & set(b_spk2utt.keys()))
    return common_spks


def main(a_dir, b_dir, c_dir):
    os.makedirs(c_dir, exist_ok=True)

    # Step 1: 合并 spk2utt
    a_spk2utt = load_spk2utt(os.path.join(a_dir, "spk2utt"))
    b_spk2utt = load_spk2utt(os.path.join(b_dir, "spk2utt"))
    common_spks = check_spk(a_spk2utt, b_spk2utt)

    if len(common_spks) != 0:
        print("Have common spk, need to check!!!")
        return

    merged_spk2utt = defaultdict(list, {k: list(v) for k, v in a_spk2utt.items()})

    for spk, utts in b_spk2utt.items():
        merged_spk2utt[spk].extend(utts)

    save_spk2utt(merged_spk2utt, os.path.join(c_dir, "spk2utt"))

    # Step 2: 合并 utt2embedding.pt
    merge_pt_dicts(
        os.path.join(a_dir, "utt2embedding.pt"),
        os.path.join(b_dir, "utt2embedding.pt"),
        os.path.join(c_dir, "utt2embedding.pt"),
    )

    # Step 3: 合并 utt2speech_token.pt
    merge_pt_dicts(
        os.path.join(a_dir, "utt2speech_token.pt"),
        os.path.join(b_dir, "utt2speech_token.pt"),
        os.path.join(c_dir, "utt2speech_token.pt"),
    )

    # Step 4: 合并 spk2embedding.pt
    merge_spk2embedding(
        os.path.join(a_dir, "spk2embedding.pt"),
        os.path.join(b_dir, "spk2embedding.pt"),
        a_spk2utt,
        b_spk2utt,
        os.path.join(c_dir, "spk2embedding.pt"),
    )
    
    cat_merge_files("text", a_dir, b_dir, c_dir)
    cat_merge_files("utt2spk", a_dir, b_dir, c_dir)
    cat_merge_files("wav.scp", a_dir, b_dir, c_dir)

    print("✅ 合并完成，保存到：", c_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--a_dir", type=str, required=True, help="原始数据目录 A")
    parser.add_argument("-b", "--b_dir", type=str, required=True, help="新增数据目录 B")
    parser.add_argument("-c", "--c_dir", type=str, required=True, help="输出目录 C")
    args = parser.parse_args()
    main(args.a_dir, args.b_dir, args.c_dir)
