import csv
from collections import defaultdict
import os
from tqdm import tqdm
from itertools import islice
from typing import Dict, List, Tuple
import shutil
import re
import jiwer

def is_yoyo(name):
    if name.count("_") == 1:
        name_no_underscore = name.replace("_", "")
        return name_no_underscore.isdigit()
    return False

def is_mucs(name):
    if name.count("_") == 2:
        name_no_underscore = name.replace("_", "")
        return name_no_underscore.isdigit()
    return False

def get_utt_from_wavpath(wav_path):
    utt = wav_path.split("/")[-1].replace("_0.wav", "").replace("_5.wav", "").replace("_11.wav", "").replace("_24.wav", "")
    return utt
    
    

def analysis(wav_paths):
    llen = len(wav_paths)
    print("NA audio count: ", llen)
    raw_audio = [get_utt_from_wavpath(wav_path) for wav_path in wav_paths if wav_path.endswith('_0.wav')]
    model5_audio = [get_utt_from_wavpath(wav_path) for wav_path in wav_paths if wav_path.endswith('_5.wav')]
    model5_in_raw = list(set(model5_audio) & set(raw_audio))
    model11_audio = [get_utt_from_wavpath(wav_path) for wav_path in wav_paths if wav_path.endswith('_11.wav')]
    model11_in_raw = list(set(model11_audio) & set(raw_audio))
    model24_audio = [get_utt_from_wavpath(wav_path) for wav_path in wav_paths if wav_path.endswith('_24.wav')]
    model24_in_raw = list(set(model24_audio) & set(raw_audio))
    print("raw_audio is NA: ", raw_audio, len(raw_audio))
    print("model 5 is NA: ", model5_audio, len(model5_audio), model5_in_raw, len(model5_in_raw))
    print("model 11 is NA: ", model11_audio, len(model11_audio), model11_in_raw, len(model11_in_raw))
    print("model 24 is NA: ", model24_audio, len(model24_audio), model24_in_raw, len(model24_in_raw))
    
    target_suffixes = {"_0.wav", "_5.wav", "_11.wav", "_24.wav"}
    # 统计每个 utt 出现的后缀
    utt_suffix_map = defaultdict(set)

    for path in wav_paths:
        filename = os.path.basename(path)
        for suffix in target_suffixes:
            if filename.endswith(suffix):
                utt = filename.replace(suffix, "")
                utt_suffix_map[utt].add(suffix)

    # 找出满足所有目标后缀的 utt
    allbad_utt = [utt for utt, suffixes in utt_suffix_map.items() if suffixes == target_suffixes]
    print("utt in all model is bad: ", allbad_utt, len(allbad_utt))


def extract_empty_text_wav_paths(csv_path: str, output_file: str=None, tar_word: str=""):
    choose_wav_paths = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            if text == tar_word:
                wav_path = row.get('wav_path', '')
                choose_wav_paths.append(wav_path)
                # print(wav_path)

    print(f'提取完成，共 {len(choose_wav_paths)} 条 text 为 {tar_word} 的 wav 路径')
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as fw:
            for choose_wav_path in choose_wav_paths:
                fw.write(choose_wav_path + "\n")
    
    return choose_wav_paths


def parse_wav_csv(csv_path: str) -> dict:
    """
    读取一个 CSV 文件，提取每一行的 wav_name 并构建对应的数据字典。

    Args:
        csv_path (str): 输入 CSV 文件路径，格式应为：wav_path,text,accuracy,realness

    Returns:
        dict: 一个字典，key 是 wav_name（去掉 .wav 的文件名），
              value 是包含 'text'、'accuracy'、'realness' 的字典。
    """
    wav_info_dict = {}

    with open(csv_path, "r", encoding="utf-8") as fr:
        reader = csv.reader(fr)
        for row in reader:
            if len(row) != 4:
                print(f"eeeeeeeeeeeeeeeeeeeeee in {row}")
                continue  # 跳过格式错误的行
            wav_path, text, accuracy, realness = row
            wav_name = wav_path.strip().split("/")[-1].replace(".wav", "")
            try:
                wav_info_dict[wav_name] = {
                    "text": text.strip(),
                    "accuracy": float(accuracy),
                    "realness": float(realness)
                }
            except ValueError:
                continue  # 跳过无法转换为 float 的行
            
    return wav_info_dict


def get_each(infile):
    yoyo_utts = set()
    tts_utts = set()
    comv_utts = set()
    indicTTS_utts = set()
    mucs_utts = set()
    vox107_utts = set()

    with open(infile, "r", encoding="utf-8") as fr:
        for line in tqdm(islice(fr, 0, None, 4), desc="Processing", unit="line"):
            wav_path = line.strip()
            utt = wav_path.split("/")[-1].replace("_0.wav", "").replace("_5.wav", "").replace("_11.wav", "").replace("_24.wav", "")
            if is_yoyo(utt):
                yoyo_utts.add(utt)
            elif utt.startswith("hi-IN-"):
                tts_utts.add(utt)
            elif utt.startswith("spk_"):
                comv_utts.add(utt)
            elif utt.startswith("indicTTS"):
                indicTTS_utts.add(utt)
            elif is_mucs(utt):
                mucs_utts.add(utt)
            else:
                vox107_utts.add(utt)      
    
    return yoyo_utts, tts_utts, comv_utts, indicTTS_utts, mucs_utts, vox107_utts


def compute_avg_metrics(raw_dict, # utt -> text
                        wav_info_dict: Dict[str, Dict[str, float]], 
                        choose_utts: List[str],
                        suffixes: List[str] = ["_0", "_5", "_11", "_24"]) -> Dict[str, Tuple[float, float]]:
    """
    给定 utt 列表，计算 <utt>_0, _5, _11, _24 的平均 accuracy 和 realness

    Args:
        wav_info_dict: 包含 accuracy 和 realness 的字典，key 为 wav_name
        choose_utts: 要计算的 utt 列表（不含后缀）

    Returns:
        一个字典，key 是 utt，value 是 (avg_accuracy, avg_realness)
    """
    print("11111111111111111111: ", len(wav_info_dict))
    print("22222222222222222222: ", len(choose_utts))
    
    results = {}
    
    for suffix in suffixes:
        accuracies = []
        realnesses = []
        gen_dict = {}
        for utt in choose_utts:
            key = f"{utt}{suffix}"
            if wav_info_dict[key]["text"] != "" and wav_info_dict[key]["text"] != "NA":
                accuracies.append(wav_info_dict[key]["accuracy"])
                realnesses.append(wav_info_dict[key]["realness"])
                gen_dict[utt] = wav_info_dict[key]["text"]
        print("3333333333333333: ", accuracies, len(accuracies))
        print("4444444444444444: ", realnesses, len(realnesses))
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_realness = sum(realnesses) / len(realnesses)
        llen = len(accuracies)
        avg_wer, avg_wer_havep = get_wer(raw_dict, gen_dict)
        results[suffix] = (llen, round(avg_accuracy, 3), round(avg_realness, 3), round(avg_wer, 3), round(avg_wer_havep, 3))
            
    return results

def copy_bad_wavs(raw_bad_files: List[str], raw_bad_dir: str):
    """
    将 raw_bad_files 中的所有 wav 文件拷贝到 raw_bad_dir 目录下。

    Args:
        raw_bad_files: 包含 wav 文件完整路径的列表。
        raw_bad_dir: 目标文件夹路径，若不存在会自动创建。
    """
    os.makedirs(raw_bad_dir, exist_ok=True)

    for wav_path in tqdm(raw_bad_files):
        if os.path.isfile(wav_path):
            filename = os.path.basename(wav_path)
            target_path = os.path.join(raw_bad_dir, filename)
            shutil.copy2(wav_path, target_path)

# raw_bad_files = ['/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/44076921_1750487496_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/78766395_1750790130_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/37969952_1750107299_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/66092250_1750703911_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/76913046_1750542288_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/43981287_1750536075_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/33219126_1750498799_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/74730792_1750865574_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/61504162_1750200199_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/58093552_1750883118_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/74760974_1750797237_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/58259577_1750787774_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/37969952_1750363679_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/78783132_1750871658_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/49896910_1750837456_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/52337304_1750776584_0.wav', '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/aa/79495964_1750523004_0.wav']
# raw_bad_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/raw_bad_audio"

# copy_bad_wavs(raw_bad_files, raw_bad_dir)



def get_wer(raw_dict, gen_dict):
    def remove_hindi_punctuation(text: str) -> str:
        # 去除指定标点符号
        # text = re.sub(r"[।॥!?.,\"'“”‘’]", "", text)
        text = re.sub(r"[।॥|｜!?.,\"'“”‘’]", "", text)  # 去掉长短竖线
        # 将多个连续空格替换成一个空格
        text = re.sub(r"\s+", " ", text)
        # 去除首尾空格
        text = text.strip()
        return text
    common_keys = list(set(raw_dict.keys()) & set(gen_dict.keys()))
    raw_texts = []
    gen_texts = []
    raw_texts_haveprounc = []
    gen_texts_haveprounc = []
    for utt in common_keys:
        raw_texts.append(remove_hindi_punctuation(raw_dict[utt]))
        gen_texts.append(remove_hindi_punctuation(gen_dict[utt]))
        raw_texts_haveprounc.append(raw_dict[utt])
        gen_texts_haveprounc.append(gen_dict[utt])
    
    wer_value = jiwer.wer(raw_texts, gen_texts)
    wer_value_havapunc = jiwer.wer(raw_texts_haveprounc, gen_texts_haveprounc)
    
    return wer_value, wer_value_havapunc
        
import json
def get_raw_dict(test_json_file):
    with open(test_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为你想要的字典格式
    converted_dict = {k: v[0] if isinstance(v, list) and v else "" for k, v in data.items()}
    return converted_dict


def main_multi():
    test_json_file = "test280.json"
    raw_dict = get_raw_dict(test_json_file)
    yoyo_utts, tts_utts, comv_utts, indicTTS_utts, mucs_utts, vox107_utts = get_each("output/output_exp/wav.scp")
    # print("yoyo_utts: ", yoyo_utts, len(yoyo_utts))  # 102
    # print("tts_utts: ", tts_utts, len(tts_utts))  # 26 
    # print("comv_utts: ", comv_utts, len(comv_utts))  # 31
    # print("indicTTS_utts: ", indicTTS_utts, len(indicTTS_utts))  # 24
    # print("mucs_utts: ", mucs_utts, len(mucs_utts))  # 0
    # print("vox107_utts: ", vox107_utts, len(vox107_utts)) # 4
    # ssum = len(yoyo_utts) + len(tts_utts) + len(comv_utts) + len(indicTTS_utts) + len(mucs_utts) + len(vox107_utts)
    # print("ssumssumssum: ", ssum)
    
    csv_path = '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/mos_results.csv'
    # output_file = '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp/mos_results_NA.txt'
    # choose_wav_paths = extract_empty_text_wav_paths(csv_path=csv_path, output_file=None, tar_word="NA")
    # analysis(choose_wav_paths)
    
    wav_info_dict = parse_wav_csv(csv_path)
    # print("1111111111111111111: ", len(wav_info_dict))
    
    dataset_dict = {"yoyo": list(yoyo_utts),
                    "tts": list(tts_utts),
                    "comv": list(comv_utts),
                    "indicTTS": list(indicTTS_utts),
                    "vox107": list(vox107_utts)}

    res = {}
    for k in dataset_dict.keys():
        data_acc_rel = compute_avg_metrics(raw_dict, wav_info_dict, dataset_dict[k])
        print("kkkkkkkkkkkkkkkkkkkkkkk: ", k)
        print("data_acc_rel: ", data_acc_rel)
        for suff in data_acc_rel.keys():
            if suff not in res:
                res[suff] = []
            res[suff].append(data_acc_rel[suff])
    
    print("rrrrrrrrrrrrrrrrrrr: ", res)
    for aa in res.keys():
        sum_acc = 0.0
        sum_rel = 0.0
        sum_count = 0.0
        sum_wer = 0.0
        sum_wer_havep = 0.0
        for ll in res[aa]:
            count, acc, rel, wer, wer_havep = ll
            sum_acc += count * acc
            sum_rel += count * rel
            sum_wer += count * wer
            sum_wer_havep += count * wer_havep
            sum_count += count
        avg_acc = sum_acc / sum_count
        avg_rel = sum_rel / sum_count
        avg_wer = sum_wer / sum_count
        avg_wer_havep = sum_wer_havep / sum_count
        print("aaaaaaaaaaaaaaaaaa: ", aa, round(avg_acc, 3), round(avg_rel, 3), round(avg_wer, 3), round(avg_wer_havep, 3))



def main():
    test_json_file = "data_yoyo_sft/test.json"
    raw_dict = get_raw_dict(test_json_file)
    csv_path = '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft/mos_results.csv'
    wavscp_path = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft/wav.scp"
    wav_info_dict = parse_wav_csv(csv_path)
    choose_utts = []
    with open(wavscp_path, "r", encoding="utf-8") as fr:
        for line in tqdm(islice(fr, 0, None, 3), desc="Processing", unit="line"):
            wav_path = line.strip()
            utt = wav_path.split("/")[-1].replace("_0.wav", "").replace("_12.wav", "").replace("_25.wav", "")
            choose_utts.append(utt)
            
    results = compute_avg_metrics(raw_dict, wav_info_dict, choose_utts, suffixes=["_0", "_12", "_25"])
    print("rrrrrrrrrrrrrrrrr: ", results)
    

def main_epoch17(suffixes):
    test_json_file = "data_yoyo_sft/test.json"
    raw_dict = get_raw_dict(test_json_file)
    choose_utts = list(raw_dict.keys())
    csv_path = '/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/bakbak/mos_results.csv'
    # wavscp_path = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft/wav.scp"
    wav_info_dict = parse_wav_csv(csv_path)            
    results = compute_avg_metrics(raw_dict, wav_info_dict, choose_utts, suffixes=suffixes)
    print("rrrrrrrrrrrrrrrrr: ", results)
    
    
if __name__ == "__main__":
    main_multi()
    main()
    # main_epoch17(suffixes=["_18"])
