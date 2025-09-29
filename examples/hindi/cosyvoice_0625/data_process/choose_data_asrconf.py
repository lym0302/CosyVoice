# coding=utf-8
import os
import argparse
import time
from tqdm import tqdm

def filter_by_asr_conf(input_file, output_file, threshold):
    kept, total = 0, 0
    kept_dur, total_dur = 0.0, 0.0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin.readlines(), desc="Filtering"):
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            try:
                dur = float(parts[3])
                conf = float(parts[4])
                total += 1
                total_dur += dur
                if conf > threshold:
                    fout.write(line)
                    kept += 1
                    kept_dur += dur
            except ValueError:
                continue

    return total, kept, total_dur, kept_dur


def main():
    parser = argparse.ArgumentParser(description="Filter audio list by ASR confidence threshold.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="input data.list")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="output file to save choose data more than asr conf thres.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for filtering")

    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    
    if not os.path.exists(input_file):
        print(f"[Error] Input file not found: {input_file}")
        return

    print(f"[Info] Filtering with threshold: {args.threshold}")
    print(f"[Info] Input: {input_file}")
    print(f"[Info] Output: {output_file}")

    st = time.time()
    total, kept, total_dur, kept_dur = filter_by_asr_conf(input_file, output_file, args.threshold)
    et = time.time()

    print(f"[Done] Total entries count: {total}, dur: {total_dur:.3f} s, {total_dur/3600:.3f} h.")
    print(f"[Done] Kept entries (> {args.threshold}) count: {kept}, dur: {kept_dur:.3f} s, {kept_dur/3600:.3f} h.")
    print(f"[Done] Time elapsed: {et - st:.2f} seconds")


if __name__ == "__main__":
    main()
