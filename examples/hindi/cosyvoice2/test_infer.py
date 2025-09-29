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
    text = 'जल्दी से दो।'
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
                
                utt = utt + "_" + spk  # for test300
                
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
    for i, j in enumerate(cosyvoice.inference_sft_nopitch(text, spk_name=spk, source_speech_token=source_speech_token, stream=False, text_frontend=False, lang='hindi')):
        audio_gen = j['tts_speech']
        torchaudio.save(outfile, audio_gen, sr)
        
def test_single_instruct(cosyvoice: CosyVoice2, text, prompt_audio_16k, outfile):
    prompt_speech_16k = load_wav(prompt_audio_16k, 16000)
    for i, j in enumerate(cosyvoice.inference_instruct2(text, 'खुशी', prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save(outfile, j['tts_speech'], cosyvoice.sample_rate)
    
     

# model_path = 'trained_models/v1_1000h-bbc_v1_240'
model_path = 'trained_models/bbc07230902_yoyo0904_thres480'
# spk_name = "33798415"
spk_name = "31036304"
cosyvoice, sr = init_cosyv2(model_path, spk_name=spk_name) # warm up done
print(f"CCCCCCCCCCCCCCCCCCcCosyVoice2 initialized with sample rate: {sr}")

# test_single(cosyvoice, "जल्दी से दो। [laughter] जल्दी से दो। [laughter] जल्दी से दो।" , "31036304", "aaa.wav")  # 不生效
prompt_audio_16k = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/enhance_output_norm_-23/31036304_ref_16k_part2_trim.wav"
test_single_instruct(cosyvoice, "जल्दी से दो।", prompt_audio_16k, "instruct_happy.wav")
exit()

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


test_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/out_spk_ref/test100.jsonl"
out_dir = "output2/zeroshot_bbc07230902_yoyo0904_thres480_avg2/aa"
speech_token_file = None
test_cosyvoice2_nopitch_infer(cosyvoice, sr, test_file, out_dir, speech_token_file)
            
