#coding=utf-8
# 根据能量阈值判断是否是真的16k 的方式不太准，先不用！
import numpy as np
import soundfile as sf
import os
import glob
import time
import argparse
from tqdm import tqdm
import shutil

def is_real_16k(wav_path, threshold=0.001):
    """
    判断是否为伪16kHz音频（实际频谱只到4kHz）。
    threshold 是高频能量所占比例的阈值，低于此值认为是假16k。
    """
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        return False  # 只判断 16kHz 音频

    # 做 FFT
    spec = np.fft.rfft(audio * np.hanning(len(audio)))
    freqs = np.fft.rfftfreq(len(audio), d=1/sr)
    power = np.abs(spec) ** 2

    # 高频能量占比
    low_freq_energy = np.sum(power[freqs <= 4000])
    high_freq_energy = np.sum(power[freqs > 4000])
    total_energy = low_freq_energy + high_freq_energy

    if total_energy == 0:
        return False  # 没有声音也当作伪音频处理

    high_freq_ratio = high_freq_energy / total_energy
    if wav_path.endswith("1750373403.wav"):
        print("1111111111111111: ", low_freq_energy, high_freq_energy, total_energy, high_freq_ratio)
    return high_freq_ratio > threshold


def process(infile, outfile, threshold, save_fake16k_dir=None):
    fake_count = 0
    fake_dur = 0.0
    total_count = 0
    total_dur = 0.0
    if save_fake16k_dir is not None:
        os.makedirs(save_fake16k_dir, exist_ok=True)
    with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding="utf-8") as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            if len(line_list) != 5:
                continue
            
            wav_path, spk, text, dur, asf_conf = line_list
            dur = float(dur)
            total_count += 1
            total_dur += dur
            
            if is_real_16k(wav_path, threshold):
                fw.write(line)
            else:
                if save_fake16k_dir is not None:
                    temp_list = wav_path.split("/")
                    spk, wavname = temp_list[-2:]
                    utt = spk + "_" + wavname
                    dst_path = f"{save_fake16k_dir}/{utt}"
                    shutil.copy2(wav_path, dst_path)
                    fake_count += 1
                    fake_dur += dur
                    
    return total_count, total_dur, fake_count, fake_dur
                
    

def main():
    parser = argparse.ArgumentParser(description="Filter audio list by ASR confidence threshold.")
    parser.add_argument("-i", "--infile", type=str, required=True, help="input data.list")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="output file to save choose data more than asr conf thres.")
    parser.add_argument("-t", "--threshold", type=float, default=0.05, help="ratio threshold")
    parser.add_argument("-s", "--save_fake_dir", type=str, default=None, help="save fake audio dir")

    args = parser.parse_args()
    
    input_file = args.infile
    output_file = args.outfile
    threshold = args.threshold
    save_fake_dir = args.save_fake_dir
    
    if not os.path.exists(input_file):
        print(f"[Error] Input file not found: {input_file}")
        return

    print(f"[Info] Filtering with threshold: {threshold}")
    print(f"[Info] Input: {input_file}")
    print(f"[Info] Output: {output_file}")

    st = time.time()
    total_count, total_dur, fake_count, fake_dur = process(input_file, output_file, threshold, save_fake_dir)
    et = time.time()

    print(f"[Done] Total entries count: {total_count}, dur: {total_dur:.3f} s, {total_dur/3600:.3f} h.")
    print(f"[Done] Fake 16k (< {threshold}) count: {fake_count}, dur: {fake_dur:.3f} s, {fake_dur/3600:.3f} h.")
    print(f"[Done] Time elapsed: {et - st:.2f} seconds")


if __name__ == "__main__":
    main()
