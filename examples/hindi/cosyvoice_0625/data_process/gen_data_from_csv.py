import os
import json
import time
import argparse
from tqdm import tqdm
from pydub import AudioSegment
import re
import pandas as pd


# def parse_iso_duration(iso_str):
#     """将 ISO 时间格式 PT1.16S 解析为秒数（浮点）"""
#     if iso_str.startswith("PT") and iso_str.endswith("S"):
#         return float(iso_str[2:-1])
#     return 0.0

import contextlib
import os
import tempfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generator, Union

import requests
from urllib.parse import urlparse


def HTTPStorage_read(url):
    # TODO @wenmeng.zwm add progress bar if file is too large
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def download_from_url(url, work_dir="/data/liangyunming/temp"):
    result = urlparse(url)
    file_path = None
    if result.scheme is not None and len(result.scheme) > 0:
        # bytes
        data = HTTPStorage_read(url)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        file_path = os.path.join(work_dir, os.path.basename(url))
        with open(file_path, "wb") as fb:
            fb.write(data)
    assert file_path is not None, f"failed to download: {url}"
    return file_path

def include_num(text):
    return bool(re.search(r"\d", text)) or "*" in text
    

def extract_audio_segment(src_path, start_sec, dur_sec, out_path):
    if src_path.startswith("http"):
        src_path = download_from_url(src_path)
    try:
        audio = AudioSegment.from_wav(src_path)
        segment = audio[start_sec * 1000 : (start_sec + dur_sec) * 1000]
        segment.export(out_path, format="wav")
        return True
    except Exception as e:
        print(f"[Error] Failed to extract segment: {e}")
        return False

def process_audio_asr(inp_csv_file, output_file, target_lang, max_dur=30):
    # base_dir = os.path.dirname(output_file)
    base_dir = os.path.dirname(os.path.abspath(output_file))
    save_audio_dir = os.path.join(base_dir, "save_audio") # 用于保存超过30s的音频
    os.makedirs(save_audio_dir, exist_ok=True)
    
    total_count = 0
    empty_count = 0
    miss_count = 0
    json_error_count = 0
    other_error_count = 0
    empty_context_count = 0
    other_lang_count = 0
    target_lang_count = 0
    
    other_lang_dur_all = 0.0
    target_lang_dur_all = 0.0
    dur_all = 0.0

    st = time.time()
    
    df = pd.read_csv(inp_csv_file, sep="\t")
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            wav_path = row["wav_file"]
            name = wav_path.split("/")[-1].replace(".wav", "")
            asr_txt_path = row["asr_file"]
            spk = row["user_id"]
            spk_audio_dir = f"{save_audio_dir}/{spk}"  # 用来存超过30s的音频
            os.makedirs(spk_audio_dir, exist_ok=True)
            total_count += 1
             
            if 1:
                if not os.path.exists(asr_txt_path):
                    asr_txt_path = asr_txt_path.replace("audio_data_realtime", "audio_data_realtime_asr")
                    if not os.path.exists(asr_txt_path):
                        print(f"[Warning] Missing ASR file: {asr_txt_path}")
                        miss_count += 1
                        continue
                
                if os.path.getsize(asr_txt_path) == 0:
                    print(f"[Warning] Empty ASR file: {asr_txt_path}")
                    empty_count += 1
                    continue

                try:
                    with open(asr_txt_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    dur_ms = data.get("durationMilliseconds", 0.0)
                    dur = dur_ms / 1000.0
                    dur_all += dur
                    rec_phrases = data.get("recognizedPhrases", [])
                    
                    if rec_phrases:
                        if dur < max_dur:
                            lang_flag = 0
                            text = ""
                            conf_avg = 0.0
                            sum_rec_list = data.get("combinedRecognizedPhrases", [])
                            if sum_rec_list:
                                text = sum_rec_list[0]["display"]
                                if include_num(text):
                                    text = sum_rec_list[0]["lexical"]
                            
                            langs, confs, text_lens = [], [], []
                            for i in range(len(rec_phrases)):
                                if "nBest" in rec_phrases[i]:
                                    lang_part = rec_phrases[i]["locale"]
                                    n_best = rec_phrases[i]["nBest"]
                                    if n_best:
                                        # text_part = n_best[0].get("display", "")
                                        text_part = n_best[0].get("lexical", "")
                                        conf_part = n_best[0].get("confidence", 0.0)
                                        langs.append(lang_part)
                                        confs.append(conf_part)
                                        text_lens.append(len(text_part.replace(" ", "")))
                            
                            if langs and all(l == target_lang for l in langs):
                                lang_flag = 1
                                # 计算加权平均置信度
                                total_len = sum(text_lens)
                                if total_len > 0:
                                    conf_avg = sum(c * l for c, l in zip(confs, text_lens)) / total_len
                                else:
                                    conf_avg = 0.0
                                
                            if lang_flag and text.strip() and conf_avg > 0.0 and dur > 0.0:
                                fout.write(f"{wav_path}\t{spk}\t{text}\t{dur}\t{conf_avg:.3f}\n")
                                target_lang_count += 1
                                target_lang_dur_all += dur
                            else:
                                other_lang_count += 1
                                other_lang_dur_all += dur
                        else:
                            lang_flag = 0
                            for i, rp in enumerate(rec_phrases):
                                if "nBest" not in rp or not rp["nBest"]:
                                    continue

                                part_text = rp["nBest"][0].get("display", "")
                                if include_num(part_text):
                                    part_text = rp["nBest"][0].get("lexical", "")
                                confidence = rp["nBest"][0].get("confidence", 0.0)
                                # offset = parse_iso_duration(rp.get("offset", "PT0S"))
                                offset = rp.get("offsetMilliseconds") / 1000.0
                                # duration = parse_iso_duration(rp.get("duration", "PT0S"))
                                duration = rp.get("durationMilliseconds") / 1000.0
                                lang = rec_phrases[i]["locale"]

                                if duration < 0.5 or not part_text.strip():
                                    continue

                                new_name = f"{name}_part{i}.wav"
                                new_path = os.path.join(spk_audio_dir, new_name)
                                if lang == target_lang and confidence > 0.0:
                                    lang_flag = 1
                                    if extract_audio_segment(wav_path, offset, duration, new_path):
                                        fout.write(f"{new_path}\t{spk}\t{part_text}\t{duration:.3f}\t{confidence:.3f}\n")
                                        target_lang_dur_all += duration
                                else:
                                    other_lang_dur_all += duration
                                    
                            if lang_flag:
                                target_lang_count += 1
                            else:
                                other_lang_count += 1
                                
                    else:
                        empty_context_count += 1
                        continue
                        
                                    
                except json.JSONDecodeError as je:
                    print(f"[Error] JSON parsing failed: {asr_txt_path} - {je}")
                    json_error_count += 1
                except Exception as e:
                    other_error_count += 1
                    print(f"[Error] Failed to process {wav_path}: {e}")

    et = time.time()

    print("====== Summary ======")
    print(f"Total files       : {total_count}, dur all: {dur_all:.3f} s, {dur_all/3600:.3f} h")
    print(f"Empty context files   : {empty_context_count}")
    print(f"Missing ASR files : {miss_count}")
    print(f"Empty ASR files   : {empty_count}")
    print(f"JSON errors       : {json_error_count}")
    print(f"Other errors      : {other_error_count}")
    
    print(f"target lang: {target_lang}, count: {target_lang_count},  dur: {target_lang_dur_all:.3f} s, {target_lang_dur_all/3600:.3f} h.")
    print(f"other lang count: {other_lang_count},  dur: {other_lang_dur_all:.3f} s, {other_lang_dur_all/3600:.3f} h.")
    
    print(f"Time elapsed      : {et - st:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Extract ASR results and align with audio paths")
    parser.add_argument("--inp_csv_file", required=True, help="inp_csv_file include wav_path and asr_path")
    parser.add_argument("--output_file", required=True, help="Output file to save results")
    parser.add_argument("--lang", type=str, default="hi-IN", help="target language")
    args = parser.parse_args()

    process_audio_asr(args.inp_csv_file, args.output_file, args.lang)

if __name__ == "__main__":
    main()
