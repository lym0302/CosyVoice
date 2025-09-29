# coding=utf-8
import os
import argparse
import soundfile as sf
from tqdm import tqdm

def save_spk_stats(stats_file, spk_dur_dict, spk_count_dict):
    with open(stats_file, 'w', encoding='utf-8') as fout:
        fout.write("spk\tcount\tdur(s)\tdur(h)\n")
        # 按 dur 降序排列
        sorted_spks = sorted(spk_dur_dict.items(), key=lambda x: x[1], reverse=True)
        for spk, dur in sorted_spks:
            count = spk_count_dict.get(spk, 0)
            fout.write(f"{spk}\t{count}\t{dur:.2f}\t{dur/3600:.3f}\n")

def get_duration(wav_path):
    """返回 wav 文件的时长（秒）"""
    try:
        with sf.SoundFile(wav_path) as f:
            duration = len(f) / f.samplerate
            return round(duration, 3)
    except Exception as e:
        print(f"❌ 无法读取音频 {wav_path}: {e}")
        return 0.0

def generate_data(transcript_path, audio_dir, output_path, dataname):
    spk_count = {}
    spk_dur = {}
    with open(transcript_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin.readlines()):
            if dataname == 'mucs':
                line_list = line.strip().split(" ")
                if len(line_list) < 2:
                    continue  # 跳过空行或格式不完整
                utt = line_list[0]               # 如 0001_030
                spk = utt.split("_")[1]          # 如 030
                text = " ".join(line_list[1:])   # 剩下是文本
                
            elif dataname == 'indicTTS':
                line_list = line.strip().split("\t")
                if len(line_list) < 2:
                    continue  # 跳过空行或格式不完整
                utt = line_list[0].replace(".txt", "")              # 如 train_hindifem_00011
                spk = utt.split("_")[1]          # hindifem
                if spk == "hindifem":
                    spk = "indicTTS_female"
                elif spk == "hindimale":
                    spk = "indicTTS_male"
                else:
                    print("eeeeeeeeeeeeeeeeerror in indicTTS spk: {spk}")
                    continue
                text = line_list[1]
                
            audio_file = os.path.join(audio_dir, spk, f"{utt}.wav")
            dur = get_duration(audio_file)
            fout.write(f"{audio_file}\t{spk}\t{text}\t{dur}\t1.0\n")
            if spk not in spk_dur.keys():
                spk_dur[spk] = 0.0
            spk_dur[spk] += dur
            if spk not in spk_count.keys():
                spk_count[spk] = 0
            spk_count[spk] += 1 

    print(f"✅ 生成完成：{output_path}")
    return spk_dur, spk_count

def main():
    parser = argparse.ArgumentParser(description="根据标注文件生成 data.list")
    parser.add_argument("--transcript", "-t", required=True, help="输入的标注文件路径")
    parser.add_argument("--audio_dir", "-a", required=True, help="音频文件所在根目录")
    parser.add_argument("--output_file", "-o", required=True, help="输出的 data.list 文件路径")
    parser.add_argument("--spkinfo_file", "-s", required=True, help="输出的 spk_info.txt 文件路径")
    parser.add_argument("--dataname", "-n", required=True, choices=['indicTTS', 'mucs'], help="数据类型")
    args = parser.parse_args()

    spk_dur, spk_count = generate_data(args.transcript, args.audio_dir, args.output_file, args.dataname)
    spkinfo_file = args.spkinfo_file
    save_spk_stats(spkinfo_file, spk_dur, spk_count)
    

if __name__ == "__main__":
    main()
