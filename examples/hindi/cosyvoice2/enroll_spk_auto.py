#coding=utf-8
import os
import torch
import sys
sys.path.append('../../../third_party/Matcha-TTS')
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
import numpy as np
import torch.nn.functional as F
import shutil
import requests
import soundfile as sf
import librosa


def front_init(model_dir):
    hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
    sample_rate = configs['sample_rate']
    return frontend, sample_rate


def get_spk2info(frontend: CosyVoiceFrontEnd, sample_rate, choose_dict, output_file, spk2embedding_file=None, save_pitch_dir=None):
    print("Enroll Spk...")
    spk2emb_dict = {}
    
    if spk2embedding_file is not None and os.path.exists(spk2embedding_file):
        spk2emb = torch.load(spk2embedding_file)
        for spk in spk2emb.keys():
            spk2emb_dict[spk] = torch.tensor(spk2emb[spk]).unsqueeze(0)
        
    spk2info_dict = {} 
    for spk, item in tqdm(choose_dict.items(), desc="Processing speakers"):
        prompt_speech_16k, prompt_text = item
        # print("11111111111111111: ", prompt_speech_16k, prompt_text)
        model_input = frontend.frontend_zero_shot('', prompt_text, load_wav(prompt_speech_16k, 16000), sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        if spk in spk2emb_dict:
            emb = spk2emb_dict[spk].to(frontend.device)
            model_input['llm_embedding'] = emb
            model_input['flow_embedding'] = emb
        model_input['prompt_pitch'] = torch.zeros(1, 0, dtype=torch.float32).to(frontend.device)
        
        if save_pitch_dir is not None:
            # prompt_pitch
            utt = spk + "_" + prompt_speech_16k.split("/")[-1].replace(".wav", "")
            f0_path = os.path.join(save_pitch_dir, f"{utt}.npy")
            if os.path.exists(f0_path):
                f0 = np.load(f0_path)
                model_input['prompt_pitch'] = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=model_input['prompt_speech_feat'].shape[1], mode='linear').view(-1).unsqueeze(0).to(frontend.device)
            else:
                print(f"Warning: Pitch file {f0_path} not found, skipping.")
            
            
        spk2info_dict[spk] = model_input
        # print("model_input: ", model_input, model_input.keys())
        
    torch.save(spk2info_dict, output_file)
    

def choose_enroll_audio(data_file):
    """
    根据 ASR 置信度选择每个说话人的 enroll audio

    参数:
        data_file (str): 每行格式为 audio_path \t spk \t text \t dur \t asr_conf

    返回:
        spk_to_enrollaudio (dict): key=spk, value=(audio_path, text)
    """
    print("Choose enroll audio according asr conf")
    spk_to_enrollaudio = {}  # key: spk, value: (audio_path, text)
    spk_to_max_conf = {}     # key: spk, value: max asr_conf

    with open(data_file, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if not line:
                continue
            try:
                audio_path, spk, text, dur, asr_conf = line.split("\t")
                asr_conf = float(asr_conf)
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue

            # 如果当前 spk 没有记录过，或者当前 asr_conf 更大，则更新
            if spk not in spk_to_max_conf or asr_conf > spk_to_max_conf[spk]:
                spk_to_max_conf[spk] = asr_conf
                spk_to_enrollaudio[spk] = (audio_path, text)

    return spk_to_enrollaudio


def download_and_resample(spk_to_enrollaudio, audio_dir_16k):
    """
    下载/拷贝音频并重采样到16kHz，直接保存到 audio_dir_16k

    参数:
        spk_to_enrollaudio (dict): {spk: (audio_path, text)}
        audio_dir_16k (str): 重采样后保存目录

    返回:
        spk_to_enrollaudio (dict): 更新后的 {spk: (audio_path_16k, text)}
    """
    print("Downoad and Resample enroll audio")
    os.makedirs(audio_dir_16k, exist_ok=True)

    for spk, (audio_path, text) in tqdm(spk_to_enrollaudio.items()):
        filename = os.path.basename(audio_path)
        output_path = os.path.join(audio_dir_16k, filename)

        # 下载或拷贝
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            if not os.path.exists(output_path):
                print(f"Downloading {audio_path} to {output_path}...")
                r = requests.get(audio_path, stream=True)
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
        else:
            if not os.path.exists(output_path):
                shutil.copy(audio_path, output_path)

        # 读取并重采样
        audio, sr = sf.read(output_path)
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sf.write(output_path, audio_16k, 16000)

        # 更新字典为绝对路径
        spk_to_enrollaudio[spk] = (os.path.abspath(output_path), text)

    return spk_to_enrollaudio

import pandas as pd

def save_spk_to_csv(spk_to_enrollaudio, csv_file):
    """
    将 spk_to_enrollaudio 保存为 CSV 文件

    参数:
        spk_to_enrollaudio (dict): {spk: (audio_path, text)}
        csv_file (str): 输出 CSV 文件路径
    """
    print(f"Save spk2enroll to csv file: {csv_file}")
    data = []
    for spk, (audio_path, text) in tqdm(spk_to_enrollaudio.items()):
        data.append([spk, audio_path, text])

    df = pd.DataFrame(data, columns=["spk", "enroll_audio_path", "text"])
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"CSV 文件已保存到: {csv_file}")

    

model_dir = "../../../pretrained_models/CosyVoice2-0.5B"
frontend, sample_rate = front_init(model_dir)



import sys
if __name__ == "__main__":
    # data_name = "bbc07230902_yoyo0904_thres480"
    data_name = sys.argv[1]
    data_file = f"filelists/{data_name}/train.list" # train.list
    # audio_dir = "datas/bbc07230902_yoyo0904_thres480/enroll/audios"
    audio_dir_16k = f"datas/{data_name}/enroll/audios_16k"
    out_spk2info_file = f"datas/{data_name}/enroll/spk2info.pt"
    # spk_to_enrollaudio = choose_enroll_audio(data_file)
    spk_to_enrollaudio = {'26172091': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/26172091/26172091_679e55a150cf410001c25f9a_1756702566.wav', 'आज अगर हम उनका साथ देंगे कल वो मेरा साथ देंगे।'), 
                          '30519799': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/30519799/30519799_5edbd66bbe34ef0001ddd89d_1751920706.wav', 'स्कूल भी जाना था घर का टेंशन?'), 
                          '31036304': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/31036304/31036304_5efc2c87d3d3030001ba6c0f_1755834785.wav', 'हाँ तो वो भी तो करना था ना, वो भी कर सकता था।'), 
                          '31398326': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/31398326/31398326_5f47abcb7524880001bf4307_1755809414.wav', 'मेरा था ही नहीं किसी से।'), 
                          '37863166': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/37863166/37863166_607c07bd865bc000016e5f63_1751706488.wav', 'इतना जल्दी मुझे खाना कौन देगा?'), 
                          '51842286': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/51842286/51842286_5efb5dbc2a590c00015fd550_1751684165.wav', 'तुम बहुत देर से इर्रिटेट कर रहे थे।'), 
                          '58974940': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/58974940/58974940_6206394928a4de0001ae5ede_1755114890.wav', 'एक साल चार महीने पहले हाँ'), 
                          '65361586': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/65361586/65361586_623ea31c1f2d180001109ca6_1753127303.wav', 'देखो घर में अगर गलती हो जाती है।'), 
                          '70326611': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/70326611/70326611_682856e8f84809000172ad4c_1751308804.wav', 'तो फिर बात होते होते एक दिन क्या हुआ अभी मतलब ठीक है तीन महीने हो गए'), 
                          '74549469': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/74549469/74549469_64a86bb81442490001cc68f1_1753216949.wav', 'इंसान को संभालना मुश्किल हो जाता है।'), 
                          '75144676': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/75144676/75144676_5eff3a0116b6330001f74601_1751311778.wav', 'उस दिन हम बाहर रख दिए थे मतलब ध्यान में नहीं था बाहर रख दिए गेट के बाहर।'), 
                          '77488385': ('https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/77488385/77488385_67c775a24050a40001b97a1d_1755013565.wav', 'हाँ, तो अभी नहीं चाहिए।')}
    spk_to_enrollaudio = download_and_resample(spk_to_enrollaudio, audio_dir_16k)
    spk2enroll_csv = f"datas/{data_name}/enroll/spk2enroll.csv"
    save_spk_to_csv(spk_to_enrollaudio, spk2enroll_csv)


    spk2embedding_file = None  # 这里如果是 sft 应该设置为 spk2embedding.pt 的路径
    save_pitch_dir = None
    get_spk2info(frontend, sample_rate, spk_to_enrollaudio, out_spk2info_file, spk2embedding_file, save_pitch_dir)
