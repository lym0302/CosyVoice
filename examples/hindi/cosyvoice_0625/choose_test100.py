import csv
import json
import random
from typing import List, Tuple

def stat_spk_utt_count(spk2utt_path: str, count_csv_path: str, top_n: int) -> List[str]:
    spk_count = {}
    with open(spk2utt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            spk, utts = parts[0], parts[1:]
            spk_count[spk] = len(utts)

    # 按 utt 数量排序，数量多的排前面
    sorted_spks = sorted(spk_count.items(), key=lambda x: x[1], reverse=True)

    # 写入 CSV 文件
    with open(count_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['spk', 'count'])
        for spk, count in sorted_spks:
            writer.writerow([spk, count])

    # 返回前 N 个说话人
    top_spks = [spk for spk, _ in sorted_spks[:top_n]]
    return top_spks

def sample_utts_from_spks(new_spk2utt_path: str, spks: List[str], utt_num_per_spk: int) -> List[str]:
    utts_selected = []
    spk2utts = {}

    with open(new_spk2utt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            spk, utts = parts[0], parts[1:]
            spk2utts[spk] = utts

    for spk in spks:
        if spk not in spk2utts:
            continue
        utts = spk2utts[spk]
        if len(utts) <= utt_num_per_spk:
            sampled = utts  # 不足则全部
        else:
            sampled = random.sample(utts, utt_num_per_spk)
        utts_selected.extend(sampled)

    return utts_selected

def build_json_from_text(text_path: str, utt_list: List[str], output_json_path: str):
    utt_set = set(utt_list)
    utt2text = {}

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', maxsplit=1)
            if len(parts) != 2:
                continue
            utt, text = parts
            if utt in utt_set:
                # 注意，json中每个utt的值是列表，存一个字符串
                utt2text[utt] = [text]

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(utt2text, f, ensure_ascii=False, indent=2)

# ==== 使用示例 ====

spk2utt_path = "data/train/spk2utt"
count_csv_path = "data/train/spk2utt_count.csv"
new_spk2utt_path = "data/test/spk2utt"
text_path = "data/test/text"
output_json_path = "test100.json"

num_choose_spk = 6  # 取前10个说话人
utt_per_spk = 10  # 每个说话人随机选10个utt

top_spks = stat_spk_utt_count(spk2utt_path, count_csv_path, num_choose_spk)
top_spks = top_spks + ["spk_f317f1d8", "spk_e4cf754e", "49896910", "48393483"]
selected_utts = sample_utts_from_spks(new_spk2utt_path, top_spks, utt_per_spk)
build_json_from_text(text_path, selected_utts, output_json_path)
