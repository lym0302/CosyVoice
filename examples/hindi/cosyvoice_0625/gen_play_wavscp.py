# coding=utf-8
import os
import random
from collections import defaultdict
import json
import pandas as pd

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
    

def get_utt2text(test_json):
    utt2text_dict = {}
    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        for utt, texts in data.items():
            if isinstance(texts, list) and texts:
                utt2text_dict[utt] = texts[0]  # 只取第一个文本
    return utt2text_dict


def gen_playlist_csv(playlists, utt2text_dict, output_file):
    records = []

    for wav_path in playlists:
        filename = os.path.basename(wav_path).replace("_18", "").replace(".wav", "")
        if filename in utt2text_dict:
            text = utt2text_dict[filename]
            records.append((wav_path, text))

    df = pd.DataFrame(records, columns=["wav_path", "text"])
    df.to_csv(output_file, index=False)
    print(f"✅ {len(records)} 条 播放路径已保存到：{output_file}")


audio_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/aa"
output_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/playlist.csv"
test_json = "data_yoyo_sft/test.json"

playlists = get_playlist(audio_dir)
utt2text_dict = get_utt2text(test_json)
gen_playlist_csv(playlists, utt2text_dict, output_file)