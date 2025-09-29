import os
import shutil
import json
from typing import Dict
from tqdm import tqdm
import argparse
import requests

def get_utt2wav(wav_scp_path: str):
    utt2wav = {}
    if wav_scp_path.endswith("wav.scp"):
        # 读取 wav.scp 映射关系
        with open(wav_scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue
                utt, path = parts
                utt2wav[utt] = path
    elif wav_scp_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(wav_scp_path)
        for idx, row in df.iterrows():
            audio_path = row['audio_path']
            spk = audio_path.split('/')[-2]
            utt = audio_path.split('/')[-1].replace('.wav', '')
            utt = spk + "_" + utt
            utt2wav[utt] = audio_path
    else:
        print(f"Do not support wav_scp_path: {wav_scp_path}")
        return {}
            
    return utt2wav


def get_utt_needed(inp_path):
    if inp_path.endswith(".json"):
        # 加载 JSON 文件，获取所需 utt 列表
        with open(inp_path, 'r', encoding='utf-8') as f:
            utt2text: Dict[str, list] = json.load(f)
        utts_needed = set(utt2text.keys())
    elif inp_path.endswith(".jsonl"):
        utts_needed = set()
        with open(inp_path, "r", encoding="utf-8") as fr:
            for line in tqdm(fr):
                record = json.loads(line)
                utt = record["utt"]
                spk = record["spk"]
                utts_needed.add(f"{spk}_{utt}")
    else:
        print(f"Do not support inp_path: {inp_path}")
        return set()
        
    return utts_needed


def copy_utts_wavs_rename(inp_path: str, wav_scp_path: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    utts_needed = get_utt_needed(inp_path)
    utt2wav = get_utt2wav(wav_scp_path)

    missing_utts = []
    nogen_utts = []
    for utt in tqdm(utts_needed):
        if utt not in utt2wav:
            print(f"[Warning] {utt} not found in wav.scp, skipped.")
            missing_utts.append(utt)
            continue
            
        src_path = utt2wav[utt]
        dst_path = os.path.join(target_dir, f"{utt}_raw.wav")

        if src_path.startswith("http"):
            try:
                r = requests.get(src_path, stream=True, timeout=30)
                if r.status_code == 200:
                    with open(dst_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                else:
                    print(f"[Warning] Download failed ({r.status_code}): {src_path}")
                    nogen_utts.append(utt)
                    continue
            except Exception as e:
                print(f"[Error] Download error for {src_path}: {e}")
                nogen_utts.append(utt)
                continue
        else:
            if not os.path.isfile(src_path):
                print(f"[Warning] File not found: {src_path}")
                missing_utts.append(utt)
                continue
            shutil.copy2(src_path, dst_path)

    print(f"\n✅ 拷贝完成：{len(utts_needed) - len(missing_utts) - len(nogen_utts)} 个文件")
    if missing_utts:
        print(f"❗ 缺失 {len(missing_utts)} 个 utt: {missing_utts}")
    if nogen_utts:
        print(f"❗ {len(nogen_utts)} 个 utt 下载失败: {nogen_utts}")


import pandas as pd
def copy_some_spk(target_spks, snr_mos_csv_file, data_file, out_data_file, save_audio_dir):
    os.makedirs(save_audio_dir, exist_ok=True)
    df = pd.read_csv(snr_mos_csv_file)
    df.columns = [c.strip() for c in df.columns]
    df['spk'] = df['audio_path'].apply(lambda x: x.split("/")[-2])
    spk2chooseaudio = {}
    for spk in target_spks:
        spk_df = df[df['spk'] == spk]
        all_audio_paths = spk_df['audio_path'].tolist()
        spk2chooseaudio[spk] = all_audio_paths
        print(f"spk {spk} has {len(all_audio_paths)} audio files")
    
        
    # 2️⃣ 合并所有 audio_path
    choose_audiopaths = set()
    for paths in spk2chooseaudio.values():
        choose_audiopaths.update(paths)
    print("len choose_audiopaths: ", len(choose_audiopaths))

    # 3️⃣ 读取 data_file，筛选并写入 out_data_file，同时复制音频
    with open(data_file, 'r', encoding='utf-8') as fin, \
         open(out_data_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            audio_path = line.split("\t")[0]
            if audio_path in choose_audiopaths:
                fout.write(line + "\n")
                # 复制音频到 save_audio_dir/spk/
                spk = audio_path.split("/")[-2]
                spk_dir = os.path.join(save_audio_dir, spk)
                os.makedirs(spk_dir, exist_ok=True)
                dst_path = os.path.join(spk_dir, os.path.basename(audio_path))
                src_path = audio_path
                if src_path.startswith("http"):
                    try:
                        r = requests.get(src_path, stream=True, timeout=30)
                        if r.status_code == 200:
                            with open(dst_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                        else:
                            print(f"[Warning] Download failed ({r.status_code}): {src_path}")

                            continue
                    except Exception as e:
                        print(f"[Error] Download error for {src_path}: {e}")
                        continue
                else:
                    if not os.path.isfile(src_path):
                        print(f"[Warning] File not found: {src_path}")
                        continue
                    shutil.copy2(src_path, dst_path)
        

# target_spks = ['31036304', '37863166']
# snr_mos_csv_file = "filelists/yoyo_20250904/snr_mos_tag_choose.csv"
# data_file = "filelists/yoyo_20250904/data.list"
# out_data_file = "filelists/yoyo_20250904_2spks/data.list"
# save_audio_dir = "filelists/yoyo_20250904_2spks/audios"
# copy_some_spk(target_spks, snr_mos_csv_file, data_file, out_data_file, save_audio_dir)
# exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default="datas/yoyo_sft/test.json", help=" test json file or jsonl file")
    parser.add_argument("-w", "--wav_scp_path", type=str, default="datas/yoyo_sft/test/wav.scp")
    parser.add_argument("-o", "--target_dir", type=str, default="output/test_yoyo30_llmavg3/all")
    args = parser.parse_args()

    inp_path = args.inp_path
    wav_scp_path = args.wav_scp_path
    target_dir = args.target_dir
    copy_utts_wavs_rename(inp_path, wav_scp_path, target_dir)