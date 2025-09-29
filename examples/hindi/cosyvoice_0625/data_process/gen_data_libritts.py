
# coding=utf-8
import os
import librosa
import argparse
from tqdm import tqdm

def get_pair(root_dir):
    print("Getting pairs...")
    wav_asr_pairs = []  # [(wav_path, asr_path)]
    for dirpath, _, filenames in os.walk(root_dir):
        # 把文件放到一个集合里，方便查找
        file_set = set(filenames)
        for fname in filenames:
            if fname.endswith(".wav"):
                base = os.path.splitext(fname)[0]  # utt1
                asr_name = base + ".normalized.txt"
                if asr_name in file_set:
                    wav_path = os.path.join(dirpath, fname)
                    asr_path = os.path.join(dirpath, asr_name)
                if os.path.exists(wav_path) and os.path.exists(asr_path):
                    wav_asr_pairs.append((wav_path, asr_path))
    return wav_asr_pairs


def get_audio_dur(wav_path):
    """获取音频时长（秒），保留三位小数"""
    try:
        dur = librosa.get_duration(path=wav_path)
        return round(dur, 3)   # 保留三位小数
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return 0.0


def get_text(asr_path):
    """读取 normalized.txt 的所有内容"""
    try:
        with open(asr_path, "r", encoding="utf-8") as fr:
            return fr.read().strip()
    except Exception as e:
        print(f"Error reading {asr_path}: {e}")
        return ""


def generate_data(root_dir, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    wav_asr_pairs = get_pair(root_dir)
    print(f"Have {len(wav_asr_pairs)} pair.")
    with open(outfile, 'w', encoding='utf-8') as fw:
        for wav_path, asr_path in tqdm(wav_asr_pairs, desc="Processing wav files"):
            dur = get_audio_dur(wav_path)
            text = get_text(asr_path)
            spk = wav_path.split("/")[-1].split("_")[0]
            fw.write(f"{wav_path}\t{spk}\t{text}\t{dur:.3f}\t1.0\n")
            

def main():
    parser = argparse.ArgumentParser(description="Generate dataset TSV from wav + normalized.txt pairs")
    parser.add_argument("-i", "--root_dir", type=str, required=True, help="Root directory containing wav and normalized.txt files")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    generate_data(args.root_dir, args.outfile)


if __name__ == "__main__":
    main()
    
