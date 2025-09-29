import os
import json
import argparse
from tqdm import tqdm
from pydub import AudioSegment
import re
import difflib
import pandas as pd


def get_itn_lexical_mapping(lexical, itn):
    lex_words = lexical.split()
    itn_words = itn.split()
    sm = difflib.SequenceMatcher(a=itn_words, b=lex_words)
    itn_to_lex_mapping = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue  # 相同的部分不用记录
        elif tag in ('replace', 'delete', 'insert'):
            # 获取对应片段
            itn_segment = " ".join(itn_words[i1:i2]) if i2 > i1 else None
            lex_segment = " ".join(lex_words[j1:j2]) if j2 > j1 else None
            if itn_segment is not None and lex_segment is not None:
                itn_to_lex_mapping.append((itn_segment, lex_segment))
    return itn_to_lex_mapping

# def restore_masked_itn_by_order(masked_itn, itn_to_lex_mapping):
#     """
#     按照 mapping 列表顺序还原 masked_itn 中的片段
#     """
#     restored = masked_itn
#     for itn_seg, lex_seg in itn_to_lex_mapping:
#         # 使用正则全词匹配
#         pattern = r'\b{}\b'.format(re.escape(itn_seg))
#         restored = re.sub(pattern, lex_seg, restored, count=1)  # 每次只替换第一个出现
#     return restored

def restore_masked_itn_by_order(masked_itn, itn_to_lex_mapping):
    """
    按照 mapping 列表顺序还原 masked_itn 中的片段
    （只要 itn_seg 存在，就替换；不再要求全词匹配）
    """
    restored = masked_itn
    for itn_seg, lex_seg in itn_to_lex_mapping:
        if itn_seg in restored:  # 如果存在，就替换
            restored = restored.replace(itn_seg, lex_seg, 1)  # 每次只替换一个
    return restored


def restore_display_from_masked(display, masked_words):
    restored = display
    for star_block, lex_seg in masked_words:
        # 替换 display 中第一个匹配的星号块
        restored = restored.replace(star_block, lex_seg, 1)
    return restored


def get_text(lexical, itn, maskedITN, display, tn=True):
    # tn: 是否进行 tn 还原, True 表示输出是文本， False 表示输出是数字
    itn_to_lex_mapping = get_itn_lexical_mapping(lexical, itn)
    # print("itn_to_lex_mapping: ", itn_to_lex_mapping)
    restore_masked_itn = restore_masked_itn_by_order(maskedITN, itn_to_lex_mapping)
    # print("restore_masked_itn: ", restore_masked_itn)
    masked_words = get_itn_lexical_mapping(lexical, restore_masked_itn)
    # print("masked_words: ", masked_words)
    # masked_words = [pair for pair in masked_words if pair[0] == "***"]  # 只保留 ***
    masked_words = [pair for pair in masked_words if "*" in pair[0]]  # 只保留包含 * 的
    restore_display = restore_display_from_masked(display, masked_words)
    if tn:
        restore_display_tn = restore_masked_itn_by_order(restore_display, itn_to_lex_mapping)
        return restore_display_tn
    return restore_display

# lexical = "अब ये मत कहना कि सौ दो सौ वाला भी नहीं हूँ मैं"
# itn = "अब ये मत कहना कि 100 200 वाला भी नहीं हूँ मैं"
# maskedITN = "अब ये मत कहना कि 100 200 वाला भी नहीं हूँ मैं।" 
# display = "अब ये मत कहना कि 100, 200 वाला भी नहीं हूँ मैं।"
# restore_display = get_text(lexical, itn, maskedITN, display)
# print("11111111111111: ", restore_display)  # 输出: 14। रंडी के बच्चे और तेरी बहन की गंदी।
# restore_display = get_text(lexical, itn, maskedITN, display, False)
# print("22222222222222: ", restore_display)  # 输出: 14। रंडी के बच्चे और तेरी बहन की गंदी।
# exit()


def choose_include_num(infile, outfile):
    with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            if len(line_list) != 5:
                print(f"error in {line}")
                continue
                
            wav_pah, spk, text, dur, asr_conf = line_list
            if bool(re.search(r'\d', text)):
                fw.write(line)
# infile = "filelists_v3/bbc_8h_240h/data_hindi.list"
# outfile = "filelists_v3/bbc_8h_240h/data_hindi_num.list"
# choose_include_num(infile, outfile)
# exit()




def merge_new(infile1, infile2, outfile):
    new = {}
    with open(infile2, "r", encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            wav_path_list = line.strip().split("\t")[0].split("/")
            nname = f"{wav_path_list[-2]}/{wav_path_list[-1]}"
            new[nname] = line
    
    with open(infile1, 'r', encoding='utf-8') as fr, open(outfile, 'w', encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            wav_path_list = line.strip().split("\t")[0].split("/")
            nname = f"{wav_path_list[-2]}/{wav_path_list[-1]}"
            if nname in new.keys():
                fw.write(new[nname])
            else:
                fw.write(line)

# infile1 = "filelists_v3/bbc_8h_240h/data_hindi.list"
# infile2 = "filelists_v3/bbc_8h_240h/data_hindi_num_new.list"
# outfile = "filelists_v3/bbc_8h_240h/data_hindi_nnew.list"       
# merge_new(infile1, infile2, outfile)
# exit()
    
    

import requests
from pydub import AudioSegment
from io import BytesIO

def extract_audio_segment(src_path, start_sec, dur_sec, out_path):
    output_dir = os.path.dirname(out_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        # http/https 链接
        if src_path.startswith("http://") or src_path.startswith("https://"):
            with requests.get(src_path, stream=True) as r:
                r.raise_for_status()
                audio_data = BytesIO(r.content)
            audio = AudioSegment.from_file(audio_data, format="wav")
        else:
            # 本地文件
            audio = AudioSegment.from_file(src_path, format="wav")

        # 截取片段
        segment = audio[start_sec * 1000 : (start_sec + dur_sec) * 1000]
        segment.export(out_path, format="wav")
        return True

    except Exception as e:
        print(f"[Error] Failed to extract segment: {e}")
        return False


def get_wav_asr_pair(input_path, asr_root=None, choose_in_list=None, choose_out_list=None):
    """
    获取 (wav_path, asr_path) 的列表
    
    参数:
        input_path: str, 可以是 csv 文件路径 或 文件夹路径
        asr_root: str, 如果 input_path 是文件夹，需要指定保存 ASR 结果的目录
        choose_in_list: 选择 wav_path 在 choose_in_list 中的
        choose_out_list: 选择 wav_path 不在 choose_out_list 中的
    
    返回:
        list of tuple: (wav_path, asr_path)
    """
    print("input_path: ", input_path)
    wav_asr_pairs = []

    if os.path.isfile(input_path) and input_path.endswith(".csv"):
        # 读取 CSV 文件
        df = pd.read_csv(input_path, sep='\t')
        if "wav_file" not in df.columns or "asr_file" not in df.columns:
            raise ValueError("CSV 文件中必须包含 'wav_file' 和 'asr_file' 两列")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            wav_path = str(row["wav_file"])
            asr_path = str(row["asr_file"])
            
            if choose_in_list is not None and wav_path not in choose_in_list:
                continue
            if choose_out_list is not None and wav_path in choose_out_list:
                continue
                
            if not os.path.exists(asr_path):
                asr_path = asr_path.replace("audio_data_realtime", "audio_data_realtime_asr")
                if not os.path.exists(asr_path):
                    print(f"[Warning] Missing ASR file: {asr_path}")
                    continue
            if os.path.getsize(asr_path) == 0:
                print(f"[Warning] Empty ASR file: {asr_path}")
                continue
            
            wav_asr_pairs.append((wav_path, asr_path))

    elif os.path.isdir(input_path):
        # 当 input_path 是包含音频的文件夹时， 必须有一个包含 asr结果的对应文件夹 asr_root
        if asr_root is None:
            raise ValueError("当 input_path 是文件夹时，必须提供 asr_root")
        for spk in tqdm(os.listdir(input_path)):
            
            spk_audio_dir = os.path.join(input_path, spk)
            spk_asr_dir = os.path.join(asr_root, spk)

            if not os.path.isdir(spk_audio_dir) or not os.path.isdir(spk_asr_dir):
                continue

            for wav_file in tqdm(os.listdir(spk_audio_dir), desc=f"{spk}", leave=False):
                if not wav_file.endswith('.wav') or "_part" in wav_file: # _part*.wav 是超过 30s 进行切片的
                    continue
                
                if choose_in_list is not None and wav_path not in choose_in_list:
                    continue
                
                if choose_out_list is not None and wav_path in choose_out_list:
                    continue

                name = wav_file.replace(".wav", "")
                asr_path = os.path.join(spk_asr_dir, f"{name}.txt")
                wav_path = os.path.join(spk_audio_dir, wav_file)                
                
                if not os.path.exists(asr_path):
                    print(f"[Warning] Missing ASR file: {asr_path}")
                    continue
                if os.path.getsize(asr_path) == 0:
                    print(f"[Warning] Empty ASR file: {asr_path}")
                    continue
                wav_asr_pairs.append((wav_path, asr_path))
    
    elif os.path.isfile(input_path) and input_path.endswith(".list"):
        bbc_dict = {"bbc_0723_0811": "/data/youtube_dataset/bbc_0723_0811/valid_rename",
                    "bbc_0827_0830": "/data/youtube_dataset/bbc_0827_0830/valid_rename",
                    "bbc_0901_0902": "/data/youtube_dataset/bbc_0901_0902/valid_rename",
                    "bbc_8h_240h": "/data/youtube_dataset/bbc_8h_240h/valid_rename",
                    }
        # 读取 list 文件
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                wav_path = line.strip().split("\t")[0]
                if not os.path.exists(wav_path):
                    print(f"[Warning] Missing WAV file: {wav_path}")
                    continue
                
                if "audio_data_realtime" in wav_path: # yoyo new
                    asr_path = wav_path.replace("audio_data_realtime", "audio_data_realtime_asr").replace(".wav", ".txt")
                
                elif "valid_rename" in wav_path: # bbc
                    asr_path = wav_path.replace("valid_rename", "asr_rename").replace(".wav", ".txt")
                    
                elif "raw_audio_part" in wav_path: # bbc part
                    wav_path_list = wav_path.split("/")
                    # bbc_name = wav_path_list[5]
                    spk = wav_path_list[-2]
                    wav_name = wav_path_list[-1]
                    wav_name = re.sub(r"_part\d+", "", wav_name)
                    # wav_dir = bbc_dict[bbc_name]
                    for k in bbc_dict.keys():
                        wav_dir = bbc_dict[k]
                        wav_path = f"{wav_dir}/{spk}/{wav_name}"
                        asr_path = wav_path.replace("valid_rename", "asr_rename").replace(".wav", ".txt")
                        if os.path.exists(wav_path) and os.path.exists(asr_path):
                            break
                    
                else: # yoyo old
                    ddir_name = wav_path.split("/")[3]
                    asr_path = wav_path.replace(f"{ddir_name}", f"{ddir_name}_asr").replace(".wav", ".txt")
                
                if not os.path.exists(asr_path):
                    print(f"[Warning] Missing ASR file: {asr_path}")
                    continue
                if os.path.getsize(asr_path) == 0:
                    print(f"[Warning] Empty ASR file: {asr_path}")
                    continue
                wav_asr_pairs.append((wav_path, asr_path))

    else:
        raise ValueError("input_path 必须是 csv 文件或文件夹路径")

    return wav_asr_pairs

def process_single_v2(wav_path, asr_txt_path, save_audio_dir, target_lang="hi-IN"):
    # 只要 len(rec_phrases) > 1 就切分, 有点细
    line = None
    lang = "hi-IN"
    duration = 0.0
    name = wav_path.split("/")[-1].replace(".wav", "")
    spk = wav_path.split("/")[-2]
    spk_audio_dir = f"{save_audio_dir}/{spk}"
    os.makedirs(spk_audio_dir, exist_ok=True)
    
    try:
        with open(asr_txt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        rec_phrases = data.get("recognizedPhrases", [])
        if rec_phrases == []:
            return None, lang, duration
            
        
        elif len(rec_phrases) == 1:  # 只有一段，不用分
            rp = rec_phrases[0]
            info = rp["nBest"][0]
            asr_conf = info.get("confidence", 0.0)
            lexical = info.get("lexical", "")
            itn = info.get("itn", "")
            maskedITN = info.get("maskedITN", "")
            display = info.get("display", "")
            offset = rp.get("offsetMilliseconds") / 1000.0
            duration = rp.get("durationMilliseconds") / 1000.0
            lang = rp.get("locale", "")
            if lang != target_lang or duration < 0.5 or asr_conf <= 0.0 or lexical == "" or itn == "" or maskedITN == "" or display == "":
                return None, lang, duration
            else:
                dur_total = data.get("durationMilliseconds", 0.0) / 1000.0
                # print("1111111111111111: ", offset, duration, dur_total)
                text = get_text(lexical, itn, maskedITN, display)
                if offset > 2.0 or offset+duration < dur_total - 2.0:  # 如果前面的静音大于 2.0 或者 最后的静音大于 2.0
                    new_name = f"{name}_part0.wav"
                    new_path = os.path.join(spk_audio_dir, new_name)
                    if extract_audio_segment(wav_path, offset, duration, new_path):
                        line = f"{new_path}\t{spk}\t{text}\t{duration:.3f}\t{asr_conf:.3f}\n"
                    
                else:
                    line = f"{wav_path}\t{spk}\t{text}\t{dur_total:.3f}\t{asr_conf:.3f}\n" 
                
        else:

                
            for i, rp in enumerate(rec_phrases):
                if "nBest" not in rp or not rp["nBest"]:
                    continue
                info = rp["nBest"][0]
                asr_conf = info.get("confidence", 0.0)
                lexical = info.get("lexical", "")
                itn = info.get("itn", "")
                maskedITN = info.get("maskedITN", "")
                display = info.get("display", "")
                offset = rp.get("offsetMilliseconds") / 1000.0
                duration = rp.get("durationMilliseconds") / 1000.0
                lang = rp.get("locale", "")
                if lang != target_lang or duration < 0.5 or asr_conf <= 0.0 or lexical == "" or itn == "" or maskedITN == "" or display == "":
                    return None, lang, duration
                else:                
                    text = get_text(lexical, itn, maskedITN, display)
                    new_name = f"{name}_part{i}.wav"
                    new_path = os.path.join(spk_audio_dir, new_name)
                    if extract_audio_segment(wav_path, offset, duration, new_path):
                        line = f"{new_path}\t{spk}\t{text}\t{duration:.3f}\t{asr_conf:.3f}\n"
                                   
    except json.JSONDecodeError as e:
        print(f"[Error] JSON parsing failed: {asr_txt_path} - {e}")
        return None, lang, duration
    except Exception as e:
        print(f"[Error] Failed to process {wav_path}: {e}")
        return None, lang, duration
        
    return line, lang, duration


def get_conf_lang(rec_phrases, target_lang="hi-IN"):
    langs, confs, text_lens = [], [], []
    for rp in rec_phrases:
        lang_part = rp.get("locale", "")
        n_best = rp.get("nBest", [])
        if n_best:
            text_part = n_best[0].get("lexical", "")
            conf_part = n_best[0].get("confidence", 0.0)
            langs.append(lang_part)
            confs.append(conf_part)
            text_lens.append(len(text_part.replace(" ", "")))

    is_target_lang = bool(langs) and all(l == target_lang for l in langs)

    # 计算加权平均置信度
    total_len = sum(text_lens)
    conf_avg = sum(c * l for c, l in zip(confs, text_lens)) / total_len if total_len > 0 else 0.0

    return conf_avg, is_target_lang


def process_single_v1(wav_path, asr_txt_path, save_audio_dir, target_lang="hi-IN", max_dur=30.0, min_dur=0.5):
    # 当 total_dur > max_dur 时，才根据 rec_phrases 进行切分， 如果此时 len(rec_phrases)==1 那也切分不了
    lines = []
    duration_total = 0.0
    tg_duration_total = 0.0
    
    name = os.path.splitext(os.path.basename(wav_path))[0]
    spk = os.path.basename(os.path.dirname(wav_path))
    spk_audio_dir = os.path.join(save_audio_dir, spk)
    # os.makedirs(spk_audio_dir, exist_ok=True)
    
    try:
        with open(asr_txt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        duration_total = data.get("durationMilliseconds", 0.0) / 1000.0
        rec_phrases = data.get("recognizedPhrases", [])
        if rec_phrases == []:
            return [], 0.0, duration_total
        
        if duration_total < min_dur:
            return [], 0.0, duration_total
        

        elif duration_total > max_dur:  # 大于 30s 根据 rec_phrases 进行切分
            for i, rp in enumerate(rec_phrases):
                n_best = rp.get("nBest", [])
                if not n_best:
                    continue
                info = rp["nBest"][0]
                lexical = info.get("lexical", "")
                itn = info.get("itn", "")
                maskedITN = info.get("maskedITN", "")
                display = info.get("display", "")
                if not all([lexical, itn, maskedITN, display]):  # 保证四者都不为 ""
                    continue
                
                lang = rp.get("locale", "")
                asr_conf = info.get("confidence", 0.0)
                if lang != target_lang or asr_conf <= 0.0:
                    continue
                
                offset = rp.get("offsetMilliseconds", 0.0) / 1000.0
                duration = rp.get("durationMilliseconds", 0.0) / 1000.0
                if duration < min_dur:
                    continue
            
                text = get_text(lexical, itn, maskedITN, display)
                new_name = f"{name}_part{i}.wav"
                new_path = os.path.join(spk_audio_dir, new_name)
                if not extract_audio_segment(wav_path, offset, duration, new_path):
                    print(f"[Warning] Failed to extract audio segment: {new_path}")
                    continue
                
                line = f"{new_path}\t{spk}\t{text}\t{duration:.3f}\t{asr_conf:.3f}\n"
                lines.append(line)
                tg_duration_total += duration
            
            
        else:
            combinedRecognizedPhrases = data.get("combinedRecognizedPhrases", [])
            if combinedRecognizedPhrases:
                info = combinedRecognizedPhrases[0]
                lexical = info.get("lexical", "")
                itn = info.get("itn", "")
                maskedITN = info.get("maskedITN", "")
                display = info.get("display", "")
                if not all([lexical, itn, maskedITN, display]):
                    return [], 0.0, duration_total
                
                asr_conf, is_target_lang = get_conf_lang(rec_phrases, target_lang=target_lang)
                if not is_target_lang or asr_conf <= 0.0:
                    return [], 0.0, duration_total
                
                text = get_text(lexical, itn, maskedITN, display)
                line = f"{wav_path}\t{spk}\t{text}\t{duration_total:.3f}\t{asr_conf:.3f}\n"
                lines.append(line)
                tg_duration_total = duration_total
                                   
    except json.JSONDecodeError as e:
        print(f"[Error] JSON parsing failed: {asr_txt_path} - {e}")
        return [], 0.0, duration_total
    except Exception as e:
        print(f"[Error] Failed to process {wav_path}: {e}")
        return [], 0.0, duration_total
        
    return lines, tg_duration_total, duration_total
    
    
def process_new(input_path, output_file, save_audio_dir, target_lang="hi-IN", asr_dir=None,
                choose_in_list=None, choose_out_list=None):
    wav_asr_pairs = get_wav_asr_pair(input_path, asr_dir, choose_in_list, choose_out_list)
    total_count = len(wav_asr_pairs)
    print(f"Total valid wav-asr pairs: {total_count}")
    tg_duration_total_all = 0.0
    duration_total_all = 0.0
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for wav_path, asr_txt_path in tqdm(wav_asr_pairs, total=len(wav_asr_pairs)):
            lines, tg_duration_total, duration_total = process_single_v1(wav_path, asr_txt_path, save_audio_dir, target_lang)
            if len(lines) > 0:
                for line in lines:
                    fout.write(line)
            tg_duration_total_all += tg_duration_total
            duration_total_all += duration_total
    
    ratio = tg_duration_total_all / duration_total_all if duration_total_all > 0 else 0.0
    
    
    print("Processing completed.")
    print(f"All: dur: {duration_total_all:.3f} s, {duration_total_all/3600:.3f} h.")          
    print(f"target lang: {target_lang}, dur: {tg_duration_total_all:.3f} s, {tg_duration_total_all/3600:.3f} h, ratio: {ratio:.3f}")
                
    

def main():
    parser = argparse.ArgumentParser(description="Extract ASR results and align with audio paths")
    parser.add_argument("-i", "--input_path", required=True, help="Root directory of audio files or csv file")
    parser.add_argument("-o", "--output_file", required=True, help="Output file to save results")
    parser.add_argument("-s", "--save_audio_dir", required=True, help="Save audio dir")
    parser.add_argument("--asr_dir", default=None, help="Root directory of ASR output (JSON format), when input_path is audio dir needed!")
    parser.add_argument("--lang", type=str, default="hi-IN", help="target language")
    
    args = parser.parse_args()
    
    output_file = args.output_file
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    choose_in_list = None
    choose_out_list = None
    
    # choose_in_list = []
    # with open("filelists_v3/yoyo_v1/data_hindi_num.list", "r", encoding='utf-8') as fr:
    #     for line in tqdm(fr.readlines()):
    #         choose_in_list.append(line.strip().split("\t")[0])


    process_new(args.input_path, args.output_file, args.save_audio_dir, args.lang, args.asr_dir,
                choose_in_list=choose_in_list, choose_out_list=choose_out_list)

if __name__ == "__main__":
    main()
