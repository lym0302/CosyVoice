import io
import torchaudio
from urllib.request import urlopen


def rread(audio_path):
    if audio_path.startswith(('http://', 'https://')):  # 如果是url
        try:
            with urlopen(audio_path) as response:
                data = response.read()
        except Exception as e:
            raise Exception(f"无法下载或处理远程音频文件 {audio_path}: {e}")
    else:
        data = open(audio_path, 'rb').read()
    return data


def read_audio(audio_path):
    if audio_path.startswith(('http://', 'https://')):  # 如果是url
        try:
            with urlopen(audio_path) as response:
                audio_bytes = response.read()
            audio_file_like = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_file_like)
        except Exception as e:
            raise Exception(f"无法下载或处理远程音频文件 {audio_path}: {e}")
    else:
        waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

# audio_url = "https://s3.ap-south-1.amazonaws.com/lc.tmp/audio_data_realtime/hi/22727707/22727707_612f5105f8499d000156e6d2_1753771145.wav"
# audio_path = "22727707_612f5105f8499d000156e6d2_1753771145.wav"
# data1 = rread(audio_url)
# data2 = rread(audio_path)
# print("data1==data2: ", data1==data2)

# waveform1, sr1 = read_audio(audio_url)
# waveform2, sr2 = read_audio(audio_path)
# print("waveform1==waveform2: ", waveform1==waveform2)
# print("sr1==sr2: ", sr1==sr2)


import re
def text_split_hindi(text):
    # 第一步：切分，排除小数点
    texts = re.split(r'(?<!\d)[।!?\.](?!\d)', text)
    texts = [s.strip() for s in texts if s.strip()]
    
    final_texts = []
    for t in texts:
        if len(t.split(" ")) > 30:  # 大于30字
            if "," in t:  # 有逗号再切
                parts = re.split(r'[,]', t)
                parts = [p.strip() for p in parts if p.strip()]
                final_texts.extend(parts)
            else:
                print(f"长句子但没有逗号: t")
                final_texts.append(t)
        else:
            final_texts.append(t)
    
    return final_texts

text = "घुड़सवारी और तीरंदाजी शामिल है वो कई भाषाओं की विद्वान थी और फ़्रेंच अंग्रेजी और उर्दू जैसी भाषाओं में भी कुशल थी रानी का विवाह शिवगंगाई के राजा से हुआ वेलु नचियार के पति मुथु वधुगनाथ पेरिया वुडिया थेवर सत्रह सौ अस्सी में ईस्ट इंडिया कंपनी के सैनिकों के साथ एक लड़ाई में मारे गए"
# final_texts = text_split_hindi(text)
# print(len(text.split(" ")))
# print(final_texts)
# print(len(final_texts))



# import re

# text = "उनको कुल्हड़ की चाय बहुत पसंद थी, जब कभी वो ट्रैन से सफर करती, उनके हमेशा फरमाइश रहती थी"

# parts_raw = re.split(r'(,)', text)
# parts = []
# i = 0
# while i < len(parts_raw):
#     p = parts_raw[i].strip()
#     # 如果下一个是逗号，则合并
#     if i + 1 < len(parts_raw) and parts_raw[i+1] == ',':
#         p = f"{p},"
#         i += 1
#     if p:
#         parts.append(p)
#     i += 1

# print(parts)



# import torch
# lora_model = "trained_models/yoyo_20250904_2spks/lora.pt"
# lora_state = torch.load(lora_model, map_location="cuda")
# for name, param_lora in lora_state.items():
#     print(name, param_lora.shape)
    
from tqdm import tqdm
def choose_include_english(infile, outfile):
    count = 0
    with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            if len(line_list) != 5:
                print(f"error in {line}")
                continue
                
            wav_pah, spk, text, dur, asr_conf = line_list
            if bool(re.search(r'[A-Za-z]', text)):
                fw.write(line)
                count += 1
                
    print(f"{count} lines save to {outfile}")
                
# infile = "filelists_v3/yoyo_v2/data_hindi.list"
# outfile = "filelists_v3/yoyo_v2/data_hindi_english.list"
# choose_include_english(infile, outfile)
# infile = "filelists_v3/yoyo_v1/data_hindi.list"
# outfile = "filelists_v3/yoyo_v1/data_hindi_english.list"
# choose_include_english(infile, outfile)
# infile = "filelists_v3/bbc_v2/data_hindi.list"
# outfile = "filelists_v3/bbc_v2/data_hindi_english.list"
# choose_include_english(infile, outfile)
# infile = "filelists/v1_1000h/data.list"
# outfile = "filelists/v1_1000h/data_english.list"
# choose_include_english(infile, outfile)
infile = "filelists_v3/bbc240h/data_hindi.list"
outfile = "filelists_v3/bbc240h/data_hindi_english.list"
choose_include_english(infile, outfile)
infile = "filelists_v3/bbc8h/data_hindi.list"
outfile = "filelists_v3/bbc8h/data_hindi_english.list"
choose_include_english(infile, outfile)
exit()
    
    