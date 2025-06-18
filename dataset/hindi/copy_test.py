# coding=utf-8
from tqdm import tqdm
import os
import shutil


file_path = "test.list"
out_dir = "test_raw"
os.makedirs(out_dir, exist_ok=True)

with open(file_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line_list = line.strip().split("\t")
        raw_path = line_list[0]
        utt = line_list[1] + "_" + line_list[0].split("/")[-1].replace(".wav", "")
        new_path = f"{out_dir}/{utt}.wav"
        
        try:
            shutil.copy(raw_path, new_path)
        except Exception as e:
            print(f"拷贝失败: {raw_path} -> {new_path}, 错误: {e}")
        