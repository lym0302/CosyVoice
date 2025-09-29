#!/usr/bin/env python3
import argparse
from tqdm import tqdm
import torchaudio
import pyworld as pw
import os
import torch
from multiprocessing import Pool
import numpy as np

# -----------------------------
# 全局变量，用于进程共享
# -----------------------------
GLOBAL_UTT2WAV = None
GLOBAL_OUT_DIR = None

def init_worker(utt2wav, out_dir):
    global GLOBAL_UTT2WAV, GLOBAL_OUT_DIR
    GLOBAL_UTT2WAV = utt2wav
    GLOBAL_OUT_DIR = out_dir

# -----------------------------
# single_job 使用全局变量
# -----------------------------
def single_job(utt, hop_size=480, sample_rate=24000):
    try:
        wav_path = GLOBAL_UTT2WAV[utt]
        
        # 加载音频
        speech, raw_sample_rate = torchaudio.load(wav_path)
        speech = speech.mean(dim=0, keepdim=True)
        if raw_sample_rate != sample_rate:
            speech = torchaudio.transforms.Resample(
                orig_freq=raw_sample_rate, new_freq=sample_rate)(speech)
        max_val = speech.abs().max()
        if max_val > 1:
            speech /= max_val
            
        waveform = speech

        # F0 提取
        frame_period = hop_size * 1000 / sample_rate
        _f0, t = pw.harvest(
            waveform.squeeze(dim=0).numpy().astype('double'),
            sample_rate,
            frame_period=frame_period
        )
        if sum(_f0 != 0) < 5:
            _f0, t = pw.dio(
                waveform.squeeze(dim=0).numpy().astype('double'),
                sample_rate,
                frame_period=frame_period
            )
        f0 = pw.stonemask(waveform.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate)
        # f0_resized = F.interpolate(
        #     torch.from_numpy(f0).float().view(1,1,-1),
        #     size=speech_feat.shape[0], mode='linear'
        # ).view(-1)

        # 保存 F0 到文件
        out_path = os.path.join(GLOBAL_OUT_DIR, f"{utt}.npy")
        # print("11111111111111111: ", f0.shape)
        np.save(out_path, f0)
        
    except Exception as e:
        print(f"Error processing {utt}: {e}")
        


def main(args):
    utt2wav = {}
    with open(f'{args.dir}/wav.scp') as f:
        for l in f:
            l = l.strip().split()
            utt2wav[l[0]] = l[1]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    processed_files = set(os.listdir(out_dir))
    utts_all = list(utt2wav.keys())
    utts = [utt for utt in utts_all if f"{utt}.npy" not in processed_files]
    print(f"{len(utts)} need to be processed.")

    # 多进程处理
    with Pool(processes=args.num_process, initializer=init_worker,
              initargs=(utt2wav, out_dir)) as pool:
        for _ in tqdm(pool.imap_unordered(single_job, utts), total=len(utts)):
            pass  #

    print(f"Finished processing {len(utts)} items. F0 tensors are saved in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True)
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    parser.add_argument("-n", "--num_process", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
