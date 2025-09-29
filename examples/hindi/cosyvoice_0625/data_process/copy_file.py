import os
import shutil
import random
import argparse
from tqdm import tqdm

def copy_wavs(infile, outdir, num_samples=-1):
    os.makedirs(outdir, exist_ok=True)

    with open(infile, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if num_samples != -1 and num_samples < len(lines):
        lines = random.sample(lines, num_samples)

    for line in tqdm(lines):
        parts = line.strip().split("\t")
        if len(parts) < 2:
            print(f"[WARNING] Skipping malformed line: {line}")
            continue

        wav_path, spk = parts[:2]
        utt = spk + "_" + os.path.basename(wav_path).replace(".wav", "")

        if not os.path.isfile(wav_path):
            print(f"[WARNING] File not found: {wav_path}")
            continue

        dst_path = os.path.join(outdir, utt + ".wav")
        shutil.copy2(wav_path, dst_path)
        print(f"[INFO] Copied: {wav_path} -> {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机拷贝音频文件到目标文件夹")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="输入的文件列表路径（每行包含音频路径和说话人）")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="输出文件夹，用于保存拷贝的音频")
    parser.add_argument("-n", "--num_samples", type=int, default=-1, help="随机拷贝的数量，-1 表示全部拷贝")

    args = parser.parse_args()
    copy_wavs(args.input_file, args.output_file, args.num_samples)
