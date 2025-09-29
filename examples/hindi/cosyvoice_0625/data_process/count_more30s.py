# coding=utf-8
from tqdm import tqdm
count = 0
total_duration = 0.0
infile = "filelists/data.list"

with open(infile, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        try:
            wavpath, spk, text, dur, asr_conf = line.strip().split("\t")
            dur = float(dur)
            if dur > 30:
                count += 1
                total_duration += dur
        except ValueError:
            print("格式错误行:", line.strip())
            continue

print(f"dur > 30s 的条数: {count}")
print(f"dur > 30s 的总时长: {total_duration:.2f} 秒")
