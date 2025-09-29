# coding=utf-8

import json
from collections import defaultdict
from tqdm import tqdm
import os
import csv
from glob import glob


def build_spk2line(jsonl_file, show_progress=True):
    """
    读取 jsonl 文件，构建 spk2line 字典
    key: spk
    value: 列表，每个元素是对应行的字典
    """
    spk2line = defaultdict(list)

    # 先统计行数，用于 tqdm
    total_lines = 0
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(jsonl_file, "r", encoding="utf-8") as f:
        iterator = f
        if show_progress:
            iterator = tqdm(f, total=total_lines, desc="Processing jsonl")

        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                spk = data.get("spk")
                if spk is not None:
                    spk2line[spk].append(data)
            except json.JSONDecodeError:
                print("JSON decode error in line:", line)

    return dict(spk2line)  # 转成普通 dict



def generate_csv_from_spk2line(spk2line, audio_folder, output_csv):
    """
    spk2line: dict, key=spk, value=list of dicts with keys: utt, text, spk
    audio_folder: 文件夹路径，里面包含 <utt>_<spk>_<tag>.wav
    output_csv: 输出 CSV 文件路径
    """
    # 先收集 audio 文件
    wav_paths = glob(os.path.join(audio_folder, "*.wav"))

    # 按 <utt>_<spk> 分组
    utt_spk_to_paths = defaultdict(list)
    for path in wav_paths:
        filename = os.path.basename(path)
        parts = filename.split("_")
        if len(parts) < 3:
            continue  # 跳过不符合命名的
        utt = parts[0]
        spk = parts[1]
        key = f"{utt}_{spk}"
        utt_spk_to_paths[key].append(path)

    # 生成 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "text"])  # 表头

        # 遍历 spk2line
        for spk, lines in spk2line.items():
            for line in lines:
                utt = line["utt"]
                key = f"{utt}_{spk}"
                text = line.get("text", "")
                # 获取对应音频文件列表
                paths = utt_spk_to_paths.get(key, [])
                for wav_path in paths:
                    writer.writerow([wav_path, text])

jsonl_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/out_spk_ref/test100.jsonl"
spk2line = build_spk2line(jsonl_file)
audio_folder = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output_test1min/to_eval/audios"
output_csv = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output_test1min/to_eval/playlist.csv"
generate_csv_from_spk2line(spk2line, audio_folder, output_csv)
