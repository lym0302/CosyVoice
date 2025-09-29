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


def json_to_jsonl(input_json_file, output_jsonl_file):
    # 读取原 JSON 文件
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        for utt, texts in tqdm(data.items(), desc="Converting JSON to JSONL"):
            spk = utt.split("_")[0]
            text = texts[0] if texts else ""
            record = {
                "utt": utt,
                "spk": spk,
                "text": text
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            
# json_to_jsonl(input_json_file="datas/yoyo_sft/test.json", 
#               output_jsonl_file="datas/yoyo_sft/test.jsonl")

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

    spk2audio = dict(zip(df["spk"], zip(df["enroll_audio_path"], df["text"])))
    return spk2audio
    


def test_infer_emo(cosyvoice: CosyVoice2, test_file, spk2refaudio_file, out_dir):
    spk2refaudio = get_spk2refaudio(spk2refaudio_file)
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
                emo_hindi = record["emo_hindi"]
                emo = record["emo"]
                
                # utt = utt + "_" + spk  # for test300
                utt = spk + "_" + utt
                
                prompt_audio_16k = spk2refaudio[spk][0]
                prompt_speech_16k = load_wav(prompt_audio_16k, 16000)
                
                for k in emo2hindi.keys():
                    emo_hindi_prompt = emo2hindi[k]
                    emo_en = k.split("/")[-1]
                    for i, j in enumerate(cosyvoice.inference_instruct2(text, emo_hindi, prompt_speech_16k, stream=False, text_frontend=False, lang='hindi')):
                        audio_gen = j['tts_speech']
                        torchaudio.save(f'{out_dir}/{utt}_{i}_{emo_en}_{emo_hindi_prompt==emo_hindi}.wav', audio_gen, sr)
                        dur += (audio_gen.shape[1] / sr)
                        count += 1
                
            except Exception as e:
                print(f"eeeeeeeeeeeeeeeeeeerror in {line} on {e}")

    et = time.time()
    infer_time = et - st
    rtf = infer_time / dur
    print(f"Count: {count}, Infer Time: {infer_time}, Dur: {dur}, RTF: {rtf}")

model_path = 'trained_models/bbc07230902_yoyo0904_thres480_emo'
spk_name = "70575863"
cosyvoice, sr = init_cosyv2(model_path, spk_name=spk_name) # warm up done
print(f"CCCCCCCCCCCCCCCCCCcCosyVoice2 initialized with sample rate: {sr}")


# text = "मेरे हिसाब से वो बंदा मालूम है।"
# prompt_audio_16k = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/datas/bbc07230902_yoyo0904_thres480/enroll/audios_16k/70575863_649e8702ec3afc000128cc99_1755014881.wav"

# for k in emo2hindi.keys():
#     emo_hindi = emo2hindi[k]
#     emo_en = k.split("/")[-1]
#     # if emo_en == "<unk>":
#     #     emo_en = "unk"
#     outfile = f"output2/emo_{emo_en}.wav"
#     test_single_instruct(cosyvoice, text, prompt_audio_16k, outfile, emo_hindi)
# exit()



## real speech token infer
# test_file = "datas/yoyo_sft/test.jsonl"
# out_dir = "output/test_yoyo30_llmavg3/real_speechtoken_nopitch/aa"
# speech_token_file = "datas/yoyo_sft/test/utt2speech_token.pt"
# test_cosyvoice2_nopitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file)

## predict speech token infer
# test_file = "datas/yoyo_sft/test.jsonl"
# out_dir = "output/test_yoyo30_llmavg3/predict_speechtoken_nopitch/aa"
# speech_token_file = None
# test_cosyvoice2_nopitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file)

import sys
# test_file = "datas/bbc07230902_yoyo0904_thres480/test30.jsonl"
test_file = "datas/bbc07230902_yoyo0904_thres480_emo/test30_emo.jsonl"
# out_dir = "output2/zeroshot_bbc07230902_yoyo0904_thres480_avg2/aa"
# out_dir = sys.argv[1]
out_dir = "output2/bbc07230902_yoyo0904_thres480_emo/bbc07230902_yoyo0904_thres480_emo_epoch2_new"
speech_token_file = None
# test_cosyvoice2_nopitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file)
spk2refaudio_file = "datas/bbc07230902_yoyo0904_thres480/enroll/spk2enroll.csv"
test_infer_emo(cosyvoice, test_file, spk2refaudio_file, out_dir)
            
