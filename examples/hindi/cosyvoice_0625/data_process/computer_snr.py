#coding=utf-8
import os
import argparse
import numpy as np
import soundfile as sf
import time
from tqdm import tqdm

def wada_snr(wav_path):
    """
    简单版 WADA-SNR 估计，基于幅值统计，针对单通道语音信号。
    返回估计的 SNR(dB)。
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    signal = np.asarray(audio)
    abs_signal = np.abs(signal)
    med = np.median(abs_signal)
    mad = np.median(np.abs(abs_signal - med))  # Median Absolute Deviation
    
    if mad == 0:
        return float('inf')
    
    noise_std = mad / 0.6745  # 估计噪声标准差（高斯假设）
    signal_std = np.std(signal)
    
    snr = 20 * np.log10(signal_std / noise_std)
    return snr


def process(infile, outfile):
    with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding="utf-8") as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            if len(line_list) != 5:
                continue
            wav_path = line_list[0]
            snr = wada_snr(wav_path)
            line_list.append(str(snr))
            fw.write("\t".join(line_list) + "\n")
    


def main():
    parser = argparse.ArgumentParser(description="Filter audio list by ASR confidence threshold.")
    parser.add_argument("-i", "--infile", type=str, required=True, help="input data.list")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="output file to save snr")

    args = parser.parse_args()
    
    input_file = args.infile
    output_file = args.outfile
    
    if not os.path.exists(input_file):
        print(f"[Error] Input file not found: {input_file}")
        return

    print(f"[Info] Input: {input_file}")
    print(f"[Info] Output: {output_file}")

    st = time.time()
    process(input_file, output_file)
    et = time.time()
    print(f"[Done] Time elapsed: {et - st:.2f} seconds")
    
if __name__ == "__main__":
    main()