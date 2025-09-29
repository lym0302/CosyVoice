# coding=utf-8
import argparse
import re
from tqdm import tqdm

def is_text_contains_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))

def process_data(input_file: str):
    total_count, total_dur = 0, 0.0
    num_count, num_dur = 0, 0.0
    keep_count, keep_dur = 0, 0.0
    keep_lines = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            wav_path, spk, text, dur_str, asr_conf = parts
            try:
                dur = float(dur_str)
            except ValueError:
                continue

            total_count += 1
            total_dur += dur

            if is_text_contains_digit(text):
                print("1111111111: ", wav_path)
                num_count += 1
                num_dur += dur
            else:
                keep_count += 1
                keep_dur += dur
                keep_lines.append(line)

    print("📊 原始数据")
    print(f"  条数: {total_count}")
    print(f"  时长: {total_dur:.2f} 秒")

    print("❌ 含数字的数据")
    print(f"  条数: {num_count}")
    print(f"  时长: {num_dur:.2f} 秒")

    print("✅ 过滤后的数据")
    print(f"  条数: {keep_count}")
    print(f"  时长: {keep_dur:.2f} 秒")
    
    return keep_lines
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤包含数字的语音样本")
    parser.add_argument("-i", "--input_file", required=True, help="输入的 data.list 文件路径")
    parser.add_argument("-o", "--output_file", required=True, help="输出的 data_rmnum.list 文件路径")
    args = parser.parse_args()

    keep_lines = process_data(args.input_file)
    with open(args.output_file, "w", encoding='utf-8') as fw:
        for line in tqdm(keep_lines):
            fw.write(line)
            
    
