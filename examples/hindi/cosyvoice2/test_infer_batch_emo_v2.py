import sys
sys.path.append('../../../third_party/Matcha-TTS')
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import time
from tqdm import tqdm
import os
from cosyvoice.cli.cosyvoice import CosyVoice2
import json
import torch
import random
from collections import defaultdict

emo2hindi = {
    "中立/neutral": "तटस्थ",       # 中立
    "开心/happy": "खुश",          # 开心
    "难过/sad": "दुखी",           # 难过
    "生气/angry": "गुस्सा",        # 生气
    "厌恶/disgusted": "घृणा",      # 厌恶
    "吃惊/surprised": "आश्चर्यचकित", # 吃惊
    "恐惧/fearful": "डर",          # 恐惧
    # "<unk>": "मिश्रित भावनाएँ",    # 未知 → 混合情感
    "其他/other": "मिश्रित भावनाएँ" # 其他 → 混合情感
}




def init_cosyv2(model_path, spk_name, 
                load_jit=False, load_trt=False, load_vllm=False, fp16=False):
    cosyvoice = CosyVoice2(model_path, load_jit=load_jit, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16)
    sr = cosyvoice.sample_rate
    # warm up
    # text = 'जल्दी से दो।'
    text = "मेरे हिसाब से वो बंदा मालूम है।"
    source_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    for i, j in enumerate(cosyvoice.inference_sft_nopitch(text, spk_name=spk_name, source_speech_token=source_speech_token, stream=False, text_frontend=False)):
        torchaudio.save('tttemp.wav', j['tts_speech'], sr)
    return cosyvoice, sr



def test_cosyvoice2_nopitch_infer(cosyvoice: CosyVoice2, sr, test_file, out_dir, speech_token_file=None):
    utt_speechtoken = {}
    if speech_token_file is not None:
        utt_speechtoken = torch.load(speech_token_file)
        
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    dur = 0.0 
    st = time.time()
    with open(test_file, "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            try:
                record = json.loads(line)
                utt = record["utt"]
                spk = record["spk"]
                text = record["text"]
                
                # utt = utt + "_" + spk  # for test300
                utt = spk + "_" + utt
                
                source_speech_token = torch.zeros(1, 0, dtype=torch.int32)  # 需要预测 speech token
                if utt_speechtoken != {} and utt in utt_speechtoken:
                    # 使用真实的 speech_token
                    source_speech_token_list = utt_speechtoken[utt]
                    source_speech_token = torch.tensor(source_speech_token_list).unsqueeze(0)
                  
                for i, j in enumerate(cosyvoice.inference_sft_nopitch(text, spk_name=spk, source_speech_token=source_speech_token, stream=False, text_frontend=False)):
                    audio_gen = j['tts_speech']
                    torchaudio.save(f'{out_dir}/{utt}_{i}.wav', audio_gen, sr)
                    dur += (audio_gen.shape[1] / sr)
                count += 1
                
            except Exception as e:
                print(f"eeeeeeeeeeeeeeeeeeerror in {line} on {e}")

    et = time.time()
    infer_time = et - st
    rtf = infer_time / dur
    print(f"Count: {count}, Infer Time: {infer_time}, Dur: {dur}, RTF: {rtf}")



def test_single(cosyvoice: CosyVoice2, text, spk, outfile):
    source_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    for i, j in enumerate(cosyvoice.inference_sft_nopitch(text, spk_name=spk, source_speech_token=source_speech_token, stream=False, text_frontend=False)):
        audio_gen = j['tts_speech']
        torchaudio.save(outfile, audio_gen, sr)
        
def test_single_instruct(cosyvoice: CosyVoice2, text, prompt_audio_16k, outfile, prompt_emo="तटस्थ"):
    prompt_speech_16k = load_wav(prompt_audio_16k, 16000)
    for i, j in enumerate(cosyvoice.inference_instruct2(text, prompt_emo, prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save(outfile, j['tts_speech'], cosyvoice.sample_rate)
        
        
import pandas as pd
def get_spk2refaudio(spk2refaudio_file):
    df = pd.read_csv(spk2refaudio_file)
    required_cols = {"spk", "enroll_audio_path", "text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 文件缺少必要列: {required_cols - set(df.columns)}")

    spk2audio = dict(zip((df["spk"].astype(str)), zip(df["enroll_audio_path"], df["text"])))
    return spk2audio
    


def test_infer_emo(cosyvoice: CosyVoice2, test_file, spk2refaudio_file, out_dir):
    spk2refaudio = get_spk2refaudio(spk2refaudio_file)
    print("spk2refaudio: ", spk2refaudio)
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    dur = 0.0 
    st = time.time()
    with open(test_file, "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            try:
                record = json.loads(line)
                utt = record["utt"]
                spk = str(record["spk"])
                text = record["text"]
                # emo_hindi = record["emo_hindi"]
                # emo = record["emo"]
                
                utt = utt + "_" + spk  # for test300
                # utt = spk + "_" + utt
                
                prompt_audio_16k = spk2refaudio[spk][0]
                prompt_speech_16k = load_wav(prompt_audio_16k, 16000)
                
                for k in emo2hindi.keys():
                    emo_hindi_prompt = emo2hindi[k]
                    emo_en = k.split("/")[-1]
                    for i, j in enumerate(cosyvoice.inference_instruct2(text, emo_hindi_prompt, prompt_speech_16k, stream=False, text_frontend=False, lang='hindi')):
                        audio_gen = j['tts_speech']
                        torchaudio.save(f'{out_dir}/{utt}_{emo_en}_{i}.wav', audio_gen, sr)
                        dur += (audio_gen.shape[1] / sr)
                        count += 1
                
            except Exception as e:
                print(f"eeeeeeeeeeeeeeeeeeerror in {line} on {e}")

    et = time.time()
    infer_time = et - st
    rtf = infer_time / dur
    print(f"Count: {count}, Infer Time: {infer_time}, Dur: {dur}, RTF: {rtf}")
    



def sample_jsonl_by_spk(input_path, output_path, n=10):
    """
    从 jsonl 文件中，每个 spk 随机抽取 n 条，保存到新的 jsonl 文件
    """
    spk_dict = defaultdict(list)

    # 读取 jsonl 文件并按 spk 分组
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                spk = data.get("spk")
                if spk is not None:
                    spk_dict[spk].append(data)
            except json.JSONDecodeError:
                print(f"[Warning] Skipping invalid JSON line: {line}")

    # 对每个 spk 随机抽取 n 条
    sampled_list = []
    for spk, items in spk_dict.items():
        if len(items) <= n:
            sampled = items
        else:
            sampled = random.sample(items, n)
        sampled_list.extend(sampled)

    # 打乱顺序
    random.shuffle(sampled_list)

    # 保存到新的 jsonl 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(sampled_list)} samples to {output_path}")


# sample_jsonl_by_spk("others/out_spk_ref/test100.jsonl", 
#                     "others/out_spk_ref/test100_choose20.jsonl", n=10)
# exit()


model_path = 'trained_models/bbc07230902_yoyo0904_thres300_emo'
spk_name = "76242429"
cosyvoice, sr = init_cosyv2(model_path, spk_name=spk_name) # warm up done
print(f"CCCCCCCCCCCCCCCCCCcCosyVoice2 initialized with sample rate: {sr}")



import sys
test_file = "others/out_spk_ref/test100_choose20.jsonl"
out_dir = "output_test1min/thres300_emo"
speech_token_file = None
# test_cosyvoice2_nopitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file)
spk2refaudio_file = "datas/bbc07230902_yoyo0904_thres480/enroll/spk2enroll_test1min.csv"
test_infer_emo(cosyvoice, test_file, spk2refaudio_file, out_dir)
            
