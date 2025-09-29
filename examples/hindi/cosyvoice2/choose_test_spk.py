# coding=utf-8
import torch
import pandas as pd
import csv

def get_in_spk(spk2emb_file):
    """
    读取 spk2embedding.pt 文件的 key, 获取训练集中的所有 spk, 返回 in_spks 列表
    """
    in_spks = []
    # 加载 PyTorch 字典
    try:
        spk2emb = torch.load(spk2emb_file, map_location='cpu')
        # 假设每个 key 是 spk
        in_spks = list(spk2emb.keys())
    except Exception as e:
        print(f"加载 {spk2emb_file} 出错: {e}")
    return in_spks


def choose_test_spk(in_spks, spkcount_file, num=10):
    """
    读取 spkcount_file, 内容是根据 total_duration 从大到小排序过的
    获取 spk 对应的 total_duration，取 total_duration < 300 并且不在 in_spks 下的前 num 个 spk
    返回 out_spks，key 是 spk, value 是 total_duration
    """
    out_spks = {}
    try:
        df = pd.read_csv(spkcount_file)
        # df = pd.read_csv(spkcount_file, sep='\t')
        # 过滤条件：total_duration < 300 且 spk 不在训练集
        filtered = df[(df['total_duration'] < 300) & (~df['spk'].isin(in_spks))]
        # 取前 num 个
        selected = filtered.head(num)
        # 转换成字典
        out_spks = dict(zip(selected['spk'], selected['total_duration']))
    except Exception as e:
        print(f"读取或处理 {spkcount_file} 出错: {e}")
    
    return out_spks


def choose_enroll_audio(data_file, test_spks, choose_csv_file, min_duration=2.0, max_duration=10.0):
    """
    根据 ASR 置信度选择每个说话人的 enroll audio

    参数:
        data_file (str): 每行格式为 audio_path \t spk \t text \t dur \t asr_conf
        test_spks (list): 待选择的测试说话人列表
        min_duration (float): 音频最短时长要求

    返回:
        spk_to_enrollaudio (dict): key=spk, value=(audio_path, text)
    """
    print("Choose enroll audio according to ASR confidence")
    spk_to_enrollaudio = {}  # key: spk, value: (audio_path, text)
    
    # 读取 choose.csv，获取所有可用 audio_path
    valid_audio_paths = set()
    with open(choose_csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get('audio_path')
            if path:
                valid_audio_paths.add(path)
    print("111111111111: ", len(valid_audio_paths))

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割字段
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            audio_path, spk, text, dur, asr_conf = parts
            if spk not in test_spks:
                continue
            if audio_path not in valid_audio_paths:
                continue
            try:
                dur = float(dur)
                asr_conf = float(asr_conf)
            except ValueError:
                continue
            if dur < min_duration or dur > max_duration:
                continue

            # 如果这个 spk 还没有选择，或者当前 asr_conf 更高，则更新
            if spk not in spk_to_enrollaudio or asr_conf > spk_to_enrollaudio[spk][2]:
                # 存储 (audio_path, text, asr_conf)
                spk_to_enrollaudio[spk] = (audio_path, text, asr_conf)
    print("spk_to_enrollaudio: ", spk_to_enrollaudio)

    # 删除 asr_conf，只保留 (audio_path, text)
    spk_to_enrollaudio = {spk: (audio, text) for spk, (audio, text, _) in spk_to_enrollaudio.items()}

    return spk_to_enrollaudio

import pandas as pd
import json
import random

def generate_test_json(snr_mos_csv_file, test_spks, data_file, output_json, num_utt_per_spk=5):
    """
    从 snr_mos_csv 中随机选择 spk_to_enrollaudio 中的 spk，每个 spk 随机 num_utt_per_spk 条记录
    保存为 JSON Lines 格式，每行一个 JSON 对象
    """
    # 1️⃣ 从 CSV 获取 spk -> 所有 audio_path
    df = pd.read_csv(snr_mos_csv_file)
    df.columns = [c.strip() for c in df.columns]
    df['spk'] = df['audio_path'].apply(lambda x: x.split("/")[-2])

    spk2chooseaudio = {}
    for spk in test_spks:
        spk_df = df[df['spk'] == spk]
        all_audio_paths = spk_df['audio_path'].tolist()
        if len(all_audio_paths) == 0:
            continue
        # 随机选择 num_utt_per_spk
        chosen_audio_paths = random.sample(all_audio_paths, min(num_utt_per_spk, len(all_audio_paths)))
        spk2chooseaudio[spk] = chosen_audio_paths
        
    # 2️⃣ 从 text 文件中获取 audio_path -> text
    audiopath_to_text = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割字段
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            audio_path, spk, text, dur, asr_conf = parts
            audiopath_to_text[audio_path] = text
    
            
    # 3️⃣ 组合 spk2chooseutt 和 utt2text 写 JSONL
    total_count = 0
    with open(output_json, 'w', encoding='utf-8') as f:
        for spk, audio_paths in spk2chooseaudio.items():
            for audio_path in audio_paths:
                utt = audio_path.split("/")[-1].replace(".wav", "")
                text = audiopath_to_text.get(audio_path, "")
                if text == "":
                    print(f"Warning: {audio_path} not found in {data_file}")
                    continue
                json_line = {"utt": utt, "spk": spk, "text": text}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                total_count += 1

    print(f"总共保存 {total_count} 条数据到 {output_json}")





if __name__ == "__main__":
    spk2emb_file1 = "datas/v1_1000h-bbc_v1_240/train/spk2embedding.pt"
    spk2emb_file2 = "datas/bbc07230902_yoyo0904_thres300/train/spk2embedding.pt"
    spkcount_file = "filelists/yoyo_20250904/snr_mos_tag_spkcount.csv"
    # spkcount_file = "filelists/yoyo_20250904/spk_info_asrconf_0.7.txt"
    

    in_spks1 = get_in_spk(spk2emb_file1)
    in_spks2 = get_in_spk(spk2emb_file2)
    in_spks = in_spks1 + in_spks2
    
    print(f"训练集中的 spk 数量: {len(in_spks)}, 其中 in_spk1: {len(in_spks1)}, in_spk2: {len(in_spks2)}")

    out_spks = choose_test_spk(in_spks, spkcount_file, num=10)
    print("选择的测试 spk:", out_spks)
    
    
    test2spks = ['31036304', '37863166']
    for ss in test2spks:
        print(f"{ss} in in_spks: {ss in in_spks}")
        
    
    test_spks = [str(x) for x in list(out_spks.keys())] + test2spks
    # test_spks = ['77488385', '51842286', '74549469', '30519799', '70326611', '75144676', '26172091', '31398326', '65361586', '58974940', '31036304', '37863166']
    print("test_spks: ", test_spks)
    data_file = "filelists/yoyo_20250904/data.list"
    choose_csv_file = "filelists/yoyo_20250904/snr_mos_tag_choose.csv"
    spk_to_enrollaudio = choose_enroll_audio(data_file, test_spks, choose_csv_file)
    print("spk_to_enrollaudio: ", spk_to_enrollaudio)
    
    output_json = "datas/tttest/test.jsonl"
    generate_test_json(choose_csv_file, test_spks, data_file, output_json, num_utt_per_spk=5)