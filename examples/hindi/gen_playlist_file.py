# coding=utf-8
import os
import random
from collections import defaultdict
import json
import pandas as pd
import argparse

def get_playlist(audio_dir):
    # 收集所有 wav 文件
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    # 分组：按前缀分，同一个前缀下的文件为一组
    utt_dict = defaultdict(list)
    for fname in wav_files:
        if "_" in fname:
            utt_prefix = fname.rsplit("_", 1)[0]
            utt_dict[utt_prefix].append(os.path.join(audio_dir, fname))

    # 打乱每组内的文件顺序
    for utt in utt_dict:
        random.shuffle(utt_dict[utt])
    # 打乱组顺序
    utt_prefixes = list(utt_dict.keys())
    random.shuffle(utt_prefixes)

    playlists = []
    for utt in utt_prefixes:
        for wav_path in utt_dict[utt]:
            playlists.append(wav_path)
            
    return playlists


def get_utt2wavpath(wav_scp):
    utt2wavpath = {}
    with open(wav_scp, "r", encoding='utf-8') as fr:
        for line in fr:
            utt, wav_path = line.strip().split(" ")
            utt2wavpath[utt] = wav_path
    return utt2wavpath
    

def get_yoyosft(text_file, wav_scp, each_spk: int = 10):
    utt2wavpath = get_utt2wavpath(wav_scp)
    
    # 临时字典：按 spk 收集 utt 和 text
    spk2utts = defaultdict(list)

    with open(text_file, "r", encoding='utf-8') as fr:
        for line in fr:
            line_list = line.strip().split(" ")
            utt = line_list[0]
            text = " ".join(line_list[1:])
            # utt, text = line.strip().split(" ")
            spk = utt.split("_")[0]
            spk2utts[spk].append((utt, text))

    utt2text_dict = {}
    wav_paths = []

    for spk, utt_text_list in spk2utts.items():
        # 打乱并选择每个spk的前each_spk个utt
        selected = random.sample(utt_text_list, min(each_spk, len(utt_text_list)))
        for utt, text in selected:
            wav_path = utt2wavpath[utt]
            if os.path.exists(wav_path):
                wav_paths.append(wav_path)
                utt2text_dict[utt] = text
            else:
                print(f"[Warning] Missing audio file: {wav_path}")

    return wav_paths, utt2text_dict


def get_data1000h(text_file, wav_scp, choose_num: int = 40):
    utt2wavpath = get_utt2wavpath(wav_scp)
    
    utt_text_list = []

    # 读取所有的 utt 和 text
    with open(text_file, "r", encoding="utf-8") as fr:
        for line in fr:
            line_list = line.strip().split(" ")
            utt = line_list[0]
            text = " ".join(line_list[1:])
            utt_text_list.append((utt, text))

    # 随机选择
    selected = random.sample(utt_text_list, min(choose_num, len(utt_text_list)))

    utt2text_dict = {}
    wav_paths = []

    for utt, text in selected:
        wav_path = utt2wavpath[utt]
        if os.path.exists(wav_path):
            wav_paths.append(wav_path)
            utt2text_dict[utt] = text
        else:
            print(f"[Warning] Missing audio file: {wav_path}")

    return wav_paths, utt2text_dict
            

def get_utt2text(test_file):
    utt2text_dict = {}
    if test_file.endswith(".json"): # 用的 test.json 那种形式
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for utt, texts in data.items():
                if isinstance(texts, list) and texts:
                    utt2text_dict[utt] = texts[0]  # 只取第一个文本
    elif test_file.endswith(".jsonl"): # 用的一行一个 json 的那种形式
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                utt = item.get("utt")
                text = item.get("text")
                if utt and text:
                    utt2text_dict[utt] = text
    else:
        print(f"eeeeeeeeeeeeeeeeerror in {test_file}")
        
    return utt2text_dict


def gen_playlist_csv(playlists, utt2text_dict, output_file):
    records = []

    for wav_path in playlists:
        utt = wav_path.split("/")[-1].replace(".wav", "")
        if utt not in utt2text_dict.keys():  
            utt = utt.split("_")[0]  # 这里可能每次规则都不太一样，根据实际情况修改！！！
            if utt not in utt2text_dict.keys():
                utt = wav_path.split("/")[-2] + "_" + wav_path.split("/")[-1].replace(".wav", "")
        if utt in utt2text_dict:
            text = utt2text_dict[utt]
            records.append((wav_path, text))
        else:
            print(f"eeeeeeeeeeerror in {wav_path}, utt: {utt}")

    df = pd.DataFrame(records, columns=["wav_path", "text"])
    df.to_csv(output_file, index=False)
    print(f"✅ {len(records)} 条 播放路径已保存到：{output_file}")



# playlists_data1000h, utt2text_dict_data1000h = get_data1000h(text_file='cosyvoice_0625/data/train/text', 
#                                                                  wav_scp ='cosyvoice_0625/data/train/wav.scp',)
# print("playlists_data1000h: ", playlists_data1000h)
# print("utt2text_dict_data1000h: ", utt2text_dict_data1000h)
# utt = playlists_data1000h[0].split("/")[-1].replace(".wav", "")
# print("111111111111111: ", utt, utt in utt2text_dict_data1000h.keys())
# exit()

def main():
    parser = argparse.ArgumentParser(description="Generate playlist CSV with reference text.")
    parser.add_argument("-a", "--audio_dir", required=True, help="Directory containing .wav files, Absolute path")
    parser.add_argument("-t", "--test_file", required=True, help="JSON file with utt_id and text")
    parser.add_argument("-o", "--output_file", required=True, help="Output CSV file path")

    args = parser.parse_args()

    audio_dir = os.path.abspath(args.audio_dir)
    playlists = get_playlist(audio_dir)
    utt2text_dict = get_utt2text(args.test_file)
    # gen_playlist_csv(playlists, utt2text_dict, args.output_file)
    playlists_yoyosft, utt2text_dict_yoyosft = get_yoyosft(text_file='cosyvoice_0625/data_yoyo_sft/train/text', 
                                                           wav_scp='cosyvoice_0625/data_yoyo_sft/train/wav.scp', )
    # print("111111111111: ", utt2text_dict_yoyosft)
    playlists_data1000h, utt2text_dict_data1000h = get_data1000h(text_file='cosyvoice_0625/data/train/text', 
                                                                 wav_scp ='cosyvoice_0625/data/train/wav.scp',)
    # print("22222222222222222: ", utt2text_dict_data1000h)
    all_playlists = playlists + playlists_yoyosft + playlists_data1000h
    random.shuffle(all_playlists)
    all_utt2text_dict = {}
    all_utt2text_dict.update(utt2text_dict)
    all_utt2text_dict.update(utt2text_dict_yoyosft)
    all_utt2text_dict.update(utt2text_dict_data1000h)
    # print(all_utt2text_dict)
    print("llllllllllllllllllen: ", len(all_playlists), len(all_utt2text_dict))
    
    gen_playlist_csv(all_playlists, all_utt2text_dict, args.output_file)
    

if __name__ == "__main__":
    main()
