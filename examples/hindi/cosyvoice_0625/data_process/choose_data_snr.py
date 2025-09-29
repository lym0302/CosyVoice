# coding=utf-8
import os
import argparse
import time
from tqdm import tqdm

def filter_by_asr_conf(input_file, output_file, asrconf_thres, snr_thres):
    kept, total = 0, 0
    kept_dur, total_dur = 0.0, 0.0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin.readlines(), desc="Filtering"):
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            try:
                dur = float(parts[3])
                conf = float(parts[4])
                snr = float(parts[5])
                total += 1
                total_dur += dur
                if conf > asrconf_thres and snr > snr_thres:
                    fout.write(line)
                    kept += 1
                    kept_dur += dur
            except ValueError:
                continue

    return total, kept, total_dur, kept_dur


def main():
    parser = argparse.ArgumentParser(description="Filter audio list by ASR confidence threshold.")
    parser.add_argument("-i", "--infile", type=str, required=True, help="input data.list")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="output file to save choose data more than asr conf thres.")
    parser.add_argument("--asr_thres", type=float, default=0.7, help="asr conf thres")
    parser.add_argument("--snr_thres", type=float, default=10, help="snr thres")

    args = parser.parse_args()
    
    input_file = args.infile
    output_file = args.outfile
    asr_thres = args.asr_thres
    snr_thres = args.snr_thres
    
    if not os.path.exists(input_file):
        print(f"[Error] Input file not found: {input_file}")
        return

    print(f"[Info] Filtering with asr conf threshold: {asr_thres}, snr threshold: {snr_thres}")
    print(f"[Info] Input: {input_file}")
    print(f"[Info] Output: {output_file}")

    st = time.time()
    total, kept, total_dur, kept_dur = filter_by_asr_conf(input_file, output_file, asr_thres, snr_thres)
    et = time.time()

    print(f"[Done] Total entries count: {total}, dur: {total_dur:.3f} s, {total_dur/3600:.3f} h.")
    print(f"[Done] Kept entries (asr_conf > {asr_thres} and snr > {snr_thres})  count: {kept}, dur: {kept_dur:.3f} s, {kept_dur/3600:.3f} h.")
    print(f"[Done] Time elapsed: {et - st:.2f} seconds")


if __name__ == "__main__":
    main()
