# coding=utf-8
import argparse
from collections import defaultdict

def stat_spk_info(input_file: str, output_file: str):
    spk_counts = defaultdict(int)
    spk_durations = defaultdict(float)

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            wav_path, spk, text, dur_str, asr_conf = parts
            try:
                duration = float(dur_str)
            except ValueError:
                continue
            spk_counts[spk] += 1
            spk_durations[spk] += duration

    # 根据总时长排序（降序）
    sorted_spks = sorted(spk_durations.items(), key=lambda x: x[1], reverse=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write("spk\tcount\ttotal_duration_sec\ttotal_duration_hour\n")
        for spk, duration in sorted_spks:
            count = spk_counts[spk]
            fout.write(f"{spk}\t{count}\t{duration:.2f}\t{duration / 3600:.2f}\n")

    print(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计每个说话人的样本数和总时长（按时长降序排序）")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="输入的 data.list 文件路径")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="输出 spk_info.txt 文件路径")

    args = parser.parse_args()
    stat_spk_info(args.input_file, args.output_file)
