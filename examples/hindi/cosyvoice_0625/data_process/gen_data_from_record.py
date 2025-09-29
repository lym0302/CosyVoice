import os
import json
import time
import argparse
from tqdm import tqdm
from pydub import AudioSegment


# def parse_iso_duration(iso_str):
#     """将 ISO 时间格式 PT1.16S 解析为秒数（浮点）"""
#     if iso_str.startswith("PT") and iso_str.endswith("S"):
#         return float(iso_str[2:-1])
#     return 0.0

def extract_audio_segment(src_path, start_sec, dur_sec, out_path):
    try:
        audio = AudioSegment.from_wav(src_path)
        segment = audio[start_sec * 1000 : (start_sec + dur_sec) * 1000]
        segment.export(out_path, format="wav")
        return True
    except Exception as e:
        print(f"[Error] Failed to extract segment: {e}")
        return False

def process_audio_asr(record_file, audio_root, asr_root, output_file, target_lang, max_dur=30):
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
    with open(output_file, 'w', encoding='utf-8') as fout:
        for spk in tqdm(os.listdir(audio_root), desc="Speakers"):
            spk_audio_dir = os.path.join(audio_root, spk)
            spk_asr_dir = os.path.join(asr_root, spk)

            if not os.path.isdir(spk_audio_dir) or not os.path.isdir(spk_asr_dir):
                continue

            for wav_file in tqdm(os.listdir(spk_audio_dir), desc=f"{spk}", leave=False):
                if not wav_file.endswith('.wav') or "_part" in wav_file: # _part*.wav 是超过 30s 进行切片的
                    continue
                total_count += 1

                name = wav_file.replace(".wav", "")
                asr_txt_path = os.path.join(spk_asr_dir, f"{name}.txt")
                wav_path = os.path.join(spk_audio_dir, wav_file)

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
                            
                            langs, confs, text_lens = [], [], []
                            for i in range(len(rec_phrases)):
                                if "nBest" in rec_phrases[i]:
                                    lang_part = rec_phrases[i]["locale"]
                                    n_best = rec_phrases[i]["nBest"]
                                    if n_best:
                                        text_part = n_best[0].get("display", "")
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
                    print(f"[Error] Failed to process {spk}/{wav_file}: {e}")

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
    parser.add_argument("--record_file", required=True, help="record have exists asr file")
    parser.add_argument("--audio_dir", required=True, help="Root directory of audio files")
    parser.add_argument("--asr_dir", required=True, help="Root directory of ASR output (JSON format)")
    parser.add_argument("--output_file", required=True, help="Output file to save results")
    parser.add_argument("--lang", type=str, default="hi-IN", help="target language")
    args = parser.parse_args()

    process_audio_asr(args.record_file, args.audio_dir, args.asr_dir, args.output_file, args.lang)

if __name__ == "__main__":
    main()
