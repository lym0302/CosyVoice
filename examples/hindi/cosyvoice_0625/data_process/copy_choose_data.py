# coding=utf-8
import os
import shutil
import argparse
from tqdm import tqdm


def copy_wavs_by_spk(data_list_path: str, new_base_dir: str):
    count = 0
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Copying files"):
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue  # 跳过格式不对的行

        wav_path, spk = parts[0], parts[1]
        wav_path = wav_path.replace("/data/open_source/VoxLingua107/hi_perspk", "/data/liangyunming/dataset/hindi/data_v2/audio_loudnorm_-16/VoxLingua107_bak")
        if not os.path.exists(wav_path):
            print(f"[WARN] File not found: {wav_path}")
            continue

        # 创建目标子文件夹
        spk_dir = os.path.join(new_base_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)

        # 拷贝到目标路径
        wav_filename = os.path.basename(wav_path)
        target_path = os.path.join(spk_dir, wav_filename)
        shutil.copy2(wav_path, target_path)
        count += 1

    print(f"✅ {count} 所有文件已拷贝完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy wav files to spk subfolders")
    parser.add_argument("-i", "--data_list_path", type=str, required=True, help="Path to data.list file")
    parser.add_argument("-o", "--new_base_dir", type=str, required=True, help="Output base directory to save copied wavs")

    args = parser.parse_args()

    copy_wavs_by_spk(args.data_list_path, args.new_base_dir)
