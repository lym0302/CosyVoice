# coding=utf-8

# coding=utf-8
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse
import re

def split_audio(file_path, inp_audio_dir, out_audio_dir):
    """
    file_path: 输入的文件，每行格式：
        utt raw_name start_time end_time
    inp_audio_dir: 原始音频目录，文件名为 {raw_name}.wav
    out_audio_dir: 切分后音频保存目录，文件名为 {utt}.wav
    """
    # 确保输出目录存在
    os.makedirs(out_audio_dir, exist_ok=True)

    with open(file_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    for line in tqdm(lines, desc="Splitting audio"):
        line = line.strip()
        if not line:
            continue
        try:
            utt, raw_name, start_time, end_time = line.split(" ")
            start_time = float(start_time)
            end_time = float(end_time)
        except Exception as e:
            print(f"Error parsing line: {line}, {e}")
            continue

        raw_audio_path = os.path.join(inp_audio_dir, f"{raw_name}.wav")
        out_audio_path = os.path.join(out_audio_dir, f"{utt}.wav")

        if not os.path.exists(raw_audio_path):
            print(f"Raw audio not found: {raw_audio_path}")
            continue

        try:
            # 加载音频
            audio, sr = librosa.load(raw_audio_path, sr=None)
            # 计算样本索引
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            # 切分
            audio_segment = audio[start_sample:end_sample]
            # 保存
            sf.write(out_audio_path, audio_segment, sr)
        except Exception as e:
            print(f"Error processing {raw_audio_path}: {e}")

def get_audio_dur(wav_path):
    """获取音频时长（秒），保留三位小数"""
    try:
        dur = librosa.get_duration(path=wav_path)
        return round(dur, 3)   # 保留三位小数
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return 0.0
            
def generate_data(text_file, out_audio_dir, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(text_file, 'r', encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split(" ")
            utt = line_list[0]
            text = " ".join(line_list[1:])
            wav_path = f"{out_audio_dir}/{utt}.wav"
            spk = utt.split("_")[0]
            if os.path.exists(wav_path):
                dur = get_audio_dur(wav_path)
                fw.write(f"{wav_path}\t{spk}\t{text}\t{dur:.3f}\t1.0\n")
    
def choose_include_num(infile, outfile):
    count = 0
    with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            if len(line_list) != 5:
                print(f"error in {line}")
                continue
                
            wav_pah, spk, text, dur, asr_conf = line_list
            if bool(re.search(r'\d', text)):
                count += 1
                fw.write(line)
    print(f"{count} line have save to {outfile}")
    
# infile = "filelists_v3/hindi_english/data_train.list"
# outfile = "filelists_v3/hindi_english/data_train_num.list"
# choose_include_num(infile, outfile)
# infile = "filelists_v3/hindi_english/data_test.list"
# outfile = "filelists_v3/hindi_english/data_test_num.list"
# choose_include_num(infile, outfile)
# exit()

def main():
    parser = argparse.ArgumentParser(description="Split audio files based on start and end times")
    parser.add_argument("-f", "--file_path", type=str, required=True, help="Path to the segments file")
    parser.add_argument("-t", "--text_file", type=str, required=True, help="Path to the segments file")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="Output data file")
    
    parser.add_argument("--inp_audio_dir", type=str, required=True, help="Directory containing raw audio files")
    parser.add_argument("--out_audio_dir", type=str, required=True, help="Directory to save split audio files")
    args = parser.parse_args()
    
    file_path = args.file_path
    inp_audio_dir = args.inp_audio_dir
    out_audio_dir = args.out_audio_dir
    outfile = args.outfile
    text_file = args.text_file
    

    split_audio(file_path, inp_audio_dir, out_audio_dir)
    generate_data(text_file, out_audio_dir, outfile)
    

if __name__ == "__main__":
    main()