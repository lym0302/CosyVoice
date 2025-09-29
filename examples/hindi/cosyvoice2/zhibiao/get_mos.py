# coding: utf-8
import pandas as pd
import argparse

def process_mos(csv_path, suffix="_0.wav"):
    """
    读取 CSV 文件，按照 filename 后缀划分 P808_MOS 列表，并计算数量和平均值。
    
    参数:
        csv_path (str): CSV 文件路径
        suffix (str): 用于筛选的文件名后缀，默认 "_0.wav"
        
    返回:
        dict: {
            "suffix_list": [...],
            "suffix_count": int,
            "suffix_mean": float,
            "other_list": [...],
            "other_count": int,
            "other_mean": float
        }
    """
    df = pd.read_csv(csv_path)
    
    # 确保 P808_MOS 为 float
    df["P808_MOS"] = df["P808_MOS"].astype(float)
    
    # 筛选后缀
    mask_suffix = df["filename"].str.endswith(suffix)
    suffix_list = df.loc[mask_suffix, "P808_MOS"].tolist()
    suffix_count = len(suffix_list)
    suffix_mean = sum(suffix_list)/suffix_count if suffix_count > 0 else 0.0
    
    # 非后缀
    mask_other = ~mask_suffix
    other_list = df.loc[mask_other, "P808_MOS"].tolist()
    other_count = len(other_list)
    other_mean = sum(other_list)/other_count if other_count > 0 else 0.0
    
    return {
        "suffix_list": suffix_list,
        "suffix_count": suffix_count,
        "suffix_mean": suffix_mean,
        "other_list": other_list,
        "other_count": other_count,
        "other_mean": other_mean
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--csv_path", type=str, required=True, help="CSV 文件路径")
    args = parser.parse_args()
    
    results = process_mos(args.csv_path, suffix="_0.wav")
    
    print(f"_0.wav 数量: {results['suffix_count']}, 平均值: {results['suffix_mean']:.3f}")
    print(f"非 _0.wav 数量: {results['other_count']}, 平均值: {results['other_mean']:.3f}")
