#coding=utf-8

import csv
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

def csv_to_dict(file_path, key_name, value_name):
    result = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = row[key_name]
            wav_path_list = wav_path.split("/")
            spk = wav_path_list[-2]
            wav_name = wav_path_list[-1].replace(".wav", "")
            if spk not in result.keys():
                result[spk] = []
            value = float(row[value_name])
            result[spk].append((wav_name, value))  
    return result

def get_dict(file_path, key_name, value_name):
    result = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = row[key_name]
            wav_path_list = wav_path.split("/")
            wav_name = wav_path_list[-1].replace(".wav", "")
            value = float(row[value_name])
            result[wav_name] = value
    return result
    


def sort_result_by_value(result):
    for spk in result:
        # 根据元组的第二个元素 value 进行降序排序
        result[spk].sort(key=lambda x: x[1], reverse=True)
    return result


def combine_and_rank(sorted_result1, sorted_result2, wavname_dur):
    combined_result = {}

    all_spks = set(sorted_result1.keys()) | set(sorted_result2.keys())

    for spk in all_spks:
        list1 = sorted_result1.get(spk, [])
        list2 = sorted_result2.get(spk, [])

        # 建立 wav_name -> (value, rank) 映射，排名从0开始
        rank_map1 = {wav_name: (value, rank) for rank, (wav_name, value) in enumerate(list1)}
        rank_map2 = {wav_name: (value, rank) for rank, (wav_name, value) in enumerate(list2)}

        # 获取两个列表 wav_name 的并集
        all_wavs = set(rank_map1.keys()) | set(rank_map2.keys())

        combined_list = []

        for wav in all_wavs:
            v1, r1 = rank_map1.get(wav, (None, float('inf')))
            v2, r2 = rank_map2.get(wav, (None, float('inf')))

            rank_sum = r1 + r2

            # value1 或 value2 可能不存在，用 None 代替
            combined_list.append((wav, v1, v2, rank_sum, wavname_dur[wav]))

        # 根据 rank_sum 排序，排名越靠前的排前面
        combined_list.sort(key=lambda x: x[3])

        # 最终只保留 (wav_name, value1, value2)
        combined_result[spk] = combined_list

    return combined_result


def save_and_get_top10(combined_result, output_file):
    top10_result = {}

    with open(output_file, 'w', encoding='utf-8') as f:
        for spk, records in combined_result.items():
            f.write(spk + "\n")
            
            # 取前10条
            top_records = records[:10]
            top10_result[spk] = top_records

            for wav_name, v1, v2, rank_num, dur in records:
                # v1, v2 可能是 None，用空字符串替代
                v1_str = str(v1) if v1 is not None else ''
                v2_str = str(v2) if v2 is not None else ''
                line = f"{spk} {wav_name} {v1_str} {v2_str} {rank_num} {float(dur)}\n"
                f.write(line)

    return top10_result

def choose(mos_csv, snr_csv, output_file):
    sorted_result1 = sort_result_by_value(csv_to_dict(file_path=mos_csv, 
            key_name = "filename", 
            value_name = "P808_MOS"))
    wavname_dur = get_dict(mos_csv, "filename", "len_in_sec")
    sorted_result2 = sort_result_by_value(csv_to_dict(snr_csv, 
            key_name = "path", 
            value_name = "snr_dB"))
    combined_result = combine_and_rank(sorted_result1, sorted_result2, wavname_dur)
    top10_result = save_and_get_top10(combined_result, output_file)
    print("top10_result: ", top10_result)
    
    


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
    spk2emb_dict = {}
    
    if spk2embedding_file is not None and os.path.exists(spk2embedding_file):
        spk2emb = torch.load(spk2embedding_file)
        for spk in spk2emb.keys():
            spk2emb_dict[spk] = torch.tensor(spk2emb[spk]).unsqueeze(0)
        
    spk2info_dict = {} 
    for spk, item in tqdm(choose_dict.items(), desc="Processing speakers"):
        prompt_speech_16k, prompt_text = item
        print("11111111111111111: ", prompt_speech_16k, prompt_text)
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
        print("model_input: ", model_input, model_input.keys())
        
    torch.save(spk2info_dict, output_file)
    


import os
import subprocess

def resample_audio_folder(input_dir, output_dir, target_rate=16000):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.isfile(input_path):
            continue  # 跳过子目录或非文件

        # 调用 ffmpeg 命令转换采样率
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-i', input_path,
            '-ar', str(target_rate),
            output_path
        ]

        print(f"Processing {input_path} -> {output_path}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# resample_audio_folder(input_dir="datas/yoyo_sft/ref_audio", 
#                       output_dir="datas/yoyo_sft/ref_audio_16k")

    

model_dir = "../../../pretrained_models/CosyVoice2-0.5B"
frontend, sample_rate = front_init(model_dir)

# choose_dict = {"33798415": ("datas/yoyo_sft/ref_audio_16k/33798415_5f10325f3fc01200010f868f_1751307359.wav", "मैं सब कुछ मैं सारा खेल खेल चुका हूँ, मुझे ये सब नहीं जानना है, मैं बहुत परेशान हूँ।"),
#                "79443944": ("datas/yoyo_sft/ref_audio_16k/79443944_68482a751ce488000177cb5b_1751555080.wav", "तो मैं बोली मैं तेरे पास नहीं जाउंगी और वैसे भी वो मुझे बदतमीज लगता है, मैं नहीं जाती उसके पास।"),
#                "65427504": ("datas/yoyo_sft/ref_audio_16k/65427504_64778a0329bc2e0001f07f8e_1751830739.wav", "और कुछ पूछना चाहते हो तो पूछ लो।"),
#                "65361586": ("datas/yoyo_sft/ref_audio_16k/65361586_683f3555e5add20001d2bc95_1751511898.wav", "ठीक है, जो रॉयल्टी के साथ चलते है ना?"),
#                "49896910": ("datas/yoyo_sft/ref_audio_16k/49896910_602bf715cf956000011ecff7_1752832506.wav", "किसी के पास चार हज़ार पांच हज़ार कोई न तो बताओ"),
#                "30629019": ("datas/yoyo_sft/ref_audio_16k/30629019_5f671d42f1b2a500012c7a75_1751744295.wav", "कोई मर्द नहीं है जीस लीड को चेस कर दे।")}
# spk2embedding_file = "datas/yoyo_sft/train/spk2embedding.pt"
# output_file = "datas/yoyo_sft/spk2info.pt"
# save_pitch_dir = "datas/yoyo_sft/f0"

# choose_dict = {"zh_female": ("../../../asset/zero_shot_prompt.wav", "希望你以后能够做得比我还好哟")}
# output_file = "trained_models/llm_snn/spk2info.pt"
# spk2embedding_file = None
# save_pitch_dir = None
# get_spk2info(frontend, sample_rate, choose_dict, output_file, spk2embedding_file, save_pitch_dir)

# choose_dict = {
#     "31036304": ("/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/out_spk_ref/aa/31036304_ref_16k_part2.wav", 
#                  "पत्तों पर टिक्की बूंद धीरे धीरे लड़खती है जैसे कोई कहानी कह रही हो दूर से आती मिट्टी की खुशबू बचपन की यादों की जगह देते हैं की यादों को जगा देते हैं कभी कभी ऐसे ही"),
#     "37863166": ("/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/out_spk_ref/aa/37863166_ref_16k_part.wav",
#                  "टिक्की बंदे धीरे धीरे लुटकते हैं जैसे कोई कहानी कह रही हो दूर से आती मिट्टी की खुशबू बचपन की यादें यादों को जगा देती है कभी कभी ऐसे ही पल में जिंदगी की रफ्तार थोड़ी धीमे बढ़ती है और दिल सोचने लगता है")}
# output_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/test2_spk2info.pt"

# choose_dict = {
#     "31036304": ("/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/enhance_output_norm_-23/31036304_ref_16k_part2_trim.wav", 
#                  "पत्तों पर टिक्की बूंद धीरे धीरे एम लड़खती है जैसे कोई कहानी कह रही हो, दूर से आती मिट्टी की खुशबू बचपन की यादों की जगह देते हैं, बचपन की यादों को जगा देते।"),
#     "37863166": ("/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/enhance_output_norm_-23/37863166_ref_16k_part_trim.wav",
#                  "टोपर टिक्की बंदे धीरे धीरे लुटकती है जैसे कोई कहानी कह रही हो। दूर से आती मिट्टी की खुशबू बचपन की यादें यादों को जगा देती है। कभी कभी ऐसे ही पल में जिंदगी की रफ्तार थोड़ी धीमे बढ़ती है।")}

choose_dict = {"Upadhyay-ceGO8lhJeAo_spk0": ("/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/datas/bbc07230902_yoyo0904_thres480/enroll/audios_16k/aaawav.wav", "इसका मकसद भारत को ग्रीन हाइड्रोजन के उत्पादन और निर्यात में वैश्विक हब बनाना है")}
output_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/trained_models/bbc07230902_yoyo0904_thres480/other_spk/onespk2info.pt"

spk2embedding_file = None
save_pitch_dir = None
get_spk2info(frontend, sample_rate, choose_dict, output_file, spk2embedding_file, save_pitch_dir)
