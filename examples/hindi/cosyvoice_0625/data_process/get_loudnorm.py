import os
import io
import csv
import librosa
import numpy as np
import pyloudnorm as pyln
from urllib.request import urlopen
from tqdm import tqdm
import pandas as pd
import argparse

# 分级阈值（LUFS）
LEVELS = [
    ("very_soft", -100, -35),  # 轻声
    ("soft", -35, -25),        # 小声
    ("normal", -25, -15),      # 正常
    ("loud", -15, -5),         # 大声
    ("very_loud", -5, 100),    # 超大声
]

def get_loudness_level(lufs: float):
    for name, low, high in LEVELS:
        if low <= lufs < high:
            return name
    return "unknown"

def load_audio(audio_path, sample_rate=16000):
    """支持本地文件或 URL"""
    if audio_path.startswith(("http://", "https://")):
        try:
            with urlopen(audio_path) as response:
                audio_bytes = response.read()
            audio_file_like = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_file_like, sr=sample_rate, mono=True)
        except Exception as e:
            raise Exception(f"无法加载远程音频 {audio_path}: {e}")
    else:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return y, sr

def get_stat(df):
    # 统计每个等级的数量和占比
    stats = df['level'].value_counts().to_frame().reset_index()
    stats.columns = ["level", "count"]
    stats["percentage"] = stats["count"] / stats["count"].sum() * 100

    # 输出整体最大值、最小值、平均值
    valid_lufs = df['loudness_LUFS'].dropna()
    if not valid_lufs.empty:
        print("\nOverall loudness statistics (LUFS):")
        print(f"Max: {valid_lufs.max():.2f}, Min: {valid_lufs.min():.2f}, Mean: {valid_lufs.mean():.2f}")

    print("\nLevel statistics:")
    print(stats)
    
    return stats

# output_csv = "filelists/yoyo_20250904/snr_mos_tag_final_300.0_loud.csv"
# df = pd.read_csv(output_csv, encoding="utf-8-sig")
# stats = get_stat(df)
# exit()


def get_ratio(output_csv, thres_list):
    """
    根据阈值列表统计三个响度等级的数量和占比

    Args:
        output_csv (str): 包含 loudness_LUFS 的 CSV 文件路径
        thres_list (list): 阈值列表 [thres_soft_normal, thres_normal_loud]

    Returns:
        pd.DataFrame: 包含 level、count、percentage
    """
    if len(thres_list) != 2:
        raise ValueError("thres_list 必须包含两个元素")

    thres_soft_normal, thres_normal_loud = thres_list

    df = pd.read_csv(output_csv)

    # 定义分类函数
    def classify_loudness(lufs):
        if lufs < thres_soft_normal:
            return "soft"
        elif lufs < thres_normal_loud:
            return "normal"
        else:
            return "loud"

    df['level'] = df['loudness_LUFS'].apply(classify_loudness)

    stats = df['level'].value_counts().to_frame().reset_index()
    stats.columns = ["level", "count"]
    stats["percentage"] = stats["count"] / stats["count"].sum() * 100

    return stats

# stats = get_ratio("filelists/bbc_0723_0811/snr_mos_tag_final_300.0_loud.csv", [-25.0, -15.0])
# print(stats)
# exit()

    

def analyze_audio_list(audio_list, output_csv):        
    results = []

    for path in tqdm(audio_list, desc="Processing audio files"):
        try:
            y, sr = load_audio(path)
            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(y)
            level = get_loudness_level(lufs)
            results.append({
                "audio_path": path,
                "loudness_LUFS": lufs,
                "level": level
            })
        except Exception as e:
            results.append({
                "audio_path": path,
                "loudness_LUFS": None,
                "level": "error"
            })
            print(f"Error processing {path}: {e}")

    # 保存 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    stats = get_stat(df)

    return df, stats

def init_line(inp_path, sep=",", name="audio_path"):
    if inp_path.endswith(".csv"):
        df = pd.read_csv(inp_path, sep=sep)  # 如果是制表符分隔，用 sep="\t"
        audio_files = df[name].tolist()
    else:  # data.list 
        audio_files = []
        with open(inp_path, 'r', encoding='utf-8') as fr:
            for line in tqdm(fr.readlines()):
                audio_files.append(line.strip().split("\t")[0])
    return audio_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process audio files for tags/SNR/MOS")
    parser.add_argument("-i", "--inp_path", type=str, required=True, help="输入文件路径, 包含音频文件的 csv 文件 或者 data.list")
    parser.add_argument("-o", "--out_csv_path", type=str, required=True, help="输出csv文件")
    # parser.add_argument("-sid", "--start_id", type=int, default=0, help="start idx")
    # parser.add_argument("-eid", "--end_id", type=int, default=9999999, help="end idx")
    
    args = parser.parse_args()
    inp_path = args.inp_path
    out_csv_path = args.out_csv_path
    
    
    audio_files = init_line(inp_path)
    df, stats = analyze_audio_list(audio_files, output_csv=out_csv_path)
