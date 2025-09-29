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
import numpy as np
import torch.nn.functional as F


def init_cosyv2_addpitch(model_path, spk_name, 
                load_jit=False, load_trt=False, load_vllm=False, fp16=False):
    cosyvoice = CosyVoice2(model_path, load_jit=load_jit, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16)
    sr = cosyvoice.sample_rate
    # warm up
    text = 'जल्दी से दो।'
    source_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    pitch=torch.zeros(1, 0, dtype=torch.float64)
    for i, j in enumerate(cosyvoice.inference_sft_addpitch(text, spk_name=spk_name, source_speech_token=source_speech_token, pitch=pitch, stream=False, text_frontend=False)):
        torchaudio.save('tttemp.wav', j['tts_speech'], sr)
    return cosyvoice, sr


def test_cosyvoice2_addpitch_infer(cosyvoice: CosyVoice2, sr, test_file, out_dir, speech_token_file=None, save_pitch_dir=None):
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
                
                source_speech_token = torch.zeros(1, 0, dtype=torch.int32)  # 需要预测 speech token
                if utt_speechtoken != {} and utt in utt_speechtoken:
                    # 使用真实的 speech_token
                    source_speech_token_list = utt_speechtoken[utt]
                    source_speech_token = torch.tensor(source_speech_token_list).unsqueeze(0)
                
                pitch=torch.zeros(1, 0, dtype=torch.float64)  # 需要预测 pitch
                if save_pitch_dir is not None:
                    # 使用真实的 pitch
                    len_pitch = len(source_speech_token_list) * 2
                    pitch_path = f"{save_pitch_dir}/{utt}.npy"
                    if not os.path.exists(pitch_path):
                        print(f"Warning: pitch file {pitch_path} not found, skipping.")
                    pitch = np.load(pitch_path)
                    pitch = F.interpolate(torch.from_numpy(pitch).view(1, 1, -1), size=len_pitch, mode='linear').view(-1).unsqueeze(0)
                  
                for i, j in enumerate(cosyvoice.inference_sft_addpitch(text, spk_name=spk, source_speech_token=source_speech_token, pitch=pitch, stream=False, text_frontend=False)):
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
    

model_path = 'trained_models/yoyo_sft_addpitch'
spk_name = "33798415"
cosyvoice, sr = init_cosyv2_addpitch(model_path, spk_name=spk_name) # warm up done
print(f"CCCCCCCCCCCCCCCCCCcCosyVoice2 initialized with sample rate: {sr}")


## real speech token  + real pitch infer
# test_file = "datas/yoyo_sft/test.jsonl"
# out_dir = "output/test_yoyo30_llmavg3/real_speechtoken_pitch/aa"
# speech_token_file = "datas/yoyo_sft/test/utt2speech_token.pt"
# save_pitch_dir = "datas/yoyo_sft/f0"
# test_cosyvoice2_addpitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file, save_pitch_dir)

# real speech token  + predict pitch infer
# test_file = "datas/yoyo_sft/test.jsonl"
# out_dir = "output/test_yoyo30_llmavg3/real_speechtoken_predict_pitch/aa"
# speech_token_file = "datas/yoyo_sft/test/utt2speech_token.pt"
# save_pitch_dir = None
# test_cosyvoice2_addpitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file, save_pitch_dir)

# predict speech token  + real pitch infer
test_file = "datas/yoyo_sft/test.jsonl"
out_dir = "output/test_yoyo30_llmavg3/predict_speechtoken_real_pitch/aa"
speech_token_file = None
save_pitch_dir = "datas/yoyo_sft/f0"
test_cosyvoice2_addpitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file, save_pitch_dir)

# predict speech token  + predict pitch infer
test_file = "datas/yoyo_sft/test.jsonl"
out_dir = "output/test_yoyo30_llmavg3/predict_speechtoken_predict_pitch/aa"
speech_token_file = None
save_pitch_dir = None
test_cosyvoice2_addpitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file, save_pitch_dir)
            
