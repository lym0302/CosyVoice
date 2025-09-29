# coding=utf-8
import re
import jiwer
import csv
from tqdm import tqdm
import os
import json

def remove_hindi_punctuation(text: str) -> str:
    return re.sub(r"[।॥!?.,\"'“”‘’]", "", text)


def get_refs(ref_csv):
    utt_text = {}
    with open(ref_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    for row in tqdm(reader, desc="Copying files", unit="file"):
        utt = row["wav_path"].split("/")[-1].replace(".wav", "")
        text = row["text"]
        utt_text[utt] = text
    return utt_text


def get_asr_text(asr_txt_path):
    text = None
    with open(asr_txt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        sum_rec_list = data.get("combinedRecognizedPhrases", [])
        if sum_rec_list:
            text = sum_rec_list[0]["lexical"]
    return text


def get_refs_hyps_pair(asr_dir, utt_text):
    raw_refs = []
    raw_hyps = []
    gen_refs = []
    gen_hyps = []
    for utt in utt_text.keys():
        asr_file = os.path.join(asr_dir, f"{utt}.txt")
        hyp = get_asr_text(asr_file)
        if hyp is not None:
            ref = remove_hindi_punctuation(utt_text[utt])
            hyp = remove_hindi_punctuation(hyp)
            if utt.endswith("_0"):
                gen_refs.append(ref)
                gen_hyps.append(hyp)
            else:
                raw_refs.append(utt_text[utt])
                raw_hyps.append(hyp)
                
    return raw_refs, raw_hyps, gen_refs, gen_hyps


def compute_detailed_wer(refs, hyps):
    """
    计算 WER 及其分项：替换、删除、插入。
    
    参数:
        refs (list[str]): 参考文本列表
        hyps (list[str]): 识别文本列表
    
    返回:
        dict: {
            "wer": float,
            "sub_wer": float,
            "del_wer": float,
            "ins_wer": float,
            "substitutions": int,
            "deletions": int,
            "insertions": int,
            "references": int
        }
    """
    measures = jiwer.compute_measures(refs, hyps)
    
    S = measures["substitutions"]
    D = measures["deletions"]
    I = measures["insertions"]
    N = sum(len(sentence) for sentence in measures["truth"])  # 总参考词数

    return {
        "wer": measures["wer"],        # 总的 WER
        "sub_wer": S / N if N > 0 else 0.0,
        "miss_extra_wer": (D + I) / N if N > 0 else 0.0,
        "del_wer": D / N if N > 0 else 0.0,
        "ins_wer": I / N if N > 0 else 0.0,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "references": N
    }


# ref_csv = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/test300_v2/playlist_400.csv"
# asr_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/test300_v2/playlist_400_asr/aa"

ref_csv = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output/yoyo_sft_zz_epoch0_test300_rename/playlist.csv"
asr_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output/yoyo_sft_zz_epoch0_test300_rename/playlist_asr/aa"

utt_text = get_refs(ref_csv)
raw_refs, raw_hyps, gen_refs, gen_hyps = get_refs_hyps_pair(asr_dir, utt_text)
if len(raw_refs) > 0 and len(raw_hyps) > 0:
    raw_res = compute_detailed_wer(raw_refs, raw_hyps)
    print("===== Raw WER Results =====")
    print(raw_res)

gen_res = compute_detailed_wer(gen_refs, gen_hyps)
print("===== Generated WER Results =====")
print(gen_res)
