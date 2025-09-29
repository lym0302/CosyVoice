# coding=utf-8
import numpy as np
import argparse
import time
from tqdm import tqdm

def compute_confidence_thresholds(input_file, output_file, split_tag="\t"):
    confidences = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Reading confidence scores"):
            parts = line.strip().split(split_tag)
            # if len(parts) != 5:
            #     continue
            try:
                conf = float(parts[-1])
                confidences.append(conf)
            except ValueError:
                continue

    if not confidences:
        print("[Error] No valid confidence scores found!")
        return

    confidences = np.array(confidences)
    percentiles = [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2]

    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write("Percentile\tConfidence_Threshold\n")
        for p in percentiles:
            threshold = np.percentile(confidences, 100 - p)  # top-p%
            fout.write(f"{p:>3}%\t\t{threshold:.4f}\n")

    print(f"[Info] Written thresholds to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute ASR confidence thresholds by percentile.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="input data.list")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="output file to save asr conf thres.")
    parser.add_argument("-s", "--split_tag", type=str, default="\t", help="split tag")
    args = parser.parse_args()

    st = time.time()
    compute_confidence_thresholds(args.input_file, args.output_file, args.split_tag)
    et = time.time()

    print(f"[Done] Time elapsed: {et - st:.2f} seconds")


if __name__ == "__main__":
    main()
