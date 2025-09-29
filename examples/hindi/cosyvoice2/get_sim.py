#coding=utf-8
import os
import torch
import sys
sys.path.append('../../../third_party/Matcha-TTS')
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
import glob


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

model_dir = "/data/liangyunming/tts_20250618/CosyVoice/pretrained_models/CosyVoice2-0.5B"
frontend, sample_rate = front_init(model_dir)

def get_spk_emb(spk2info_path):
    spk_emb_dict = {}
    spk2info = torch.load(spk2info_path)
    
    spk2info_name = spk2info_path.split("/")[-1]
    if "spk2info" in spk2info_name:
        for spk, info in spk2info.items():
            spk_emb_dict[spk] = info['flow_embedding']
    elif "spk2emb" in spk2info_name:
        for spk in spk2info.keys():
            spk_emb_dict[spk] = torch.tensor(spk2info[spk], dtype=torch.float32).unsqueeze(0)
    else:
        print(f"Do not support {spk2info_path}")
    return spk_emb_dict


def get_embedding(wav_path):
    embedding = frontend._extract_spk_embedding(load_wav(wav_path, 16000))
    return embedding


# def gen_pair(wav_dir, spk_emb_dict):
#     ref_embs = []
#     gen_embs = []
#     # wav_paths = glob.glob(os.path.join(wav_dir, "*_0.wav"))
#     wav_paths = glob.glob(os.path.join(wav_dir, "*.wav"))
#     for wav_path in tqdm(wav_paths):
#         utt = wav_path.split("/")[-1].replace(".wav", "")
#         spk = utt.split("_")[1]
#         if spk not in spk_emb_dict:
#             print(f"Speaker {spk} not found in spk_emb_dict, skipping {wav_path}.")
#             continue
#         ref_embs.append(spk_emb_dict[spk].to(frontend.device))
#         gen_embs.append(get_embedding(wav_path).to(frontend.device))
    
#     if len(ref_embs) == 0:
#         return None, None  # 没有有效数据

#     ref_embs_tensor = torch.cat(ref_embs, dim=0)   # [N, 192]
#     gen_embs_tensor = torch.cat(gen_embs, dim=0)   # [N, 192]
    
#     return ref_embs_tensor, gen_embs_tensor

def gen_pair(wav_dir, spk_emb_dict):
    spk_dict = {}  # 存储每个 spk 的 ref 和 gen embedding 列表
    
    wav_paths = glob.glob(os.path.join(wav_dir, "*.wav"))
    for wav_path in tqdm(wav_paths):
        utt = os.path.basename(wav_path).replace(".wav", "")
        spk = utt.split("_")[1]  # 提取 spk
        
        if spk not in spk_emb_dict:
            print(f"Speaker {spk} not found in spk_emb_dict, skipping {wav_path}.")
            continue
        
        # 初始化该 spk 的列表
        if spk not in spk_dict:
            spk_dict[spk] = {"ref_embs": [], "gen_embs": []}
        
        spk_dict[spk]["ref_embs"].append(spk_emb_dict[spk].to(frontend.device))
        spk_dict[spk]["gen_embs"].append(get_embedding(wav_path).to(frontend.device))
    
    # 把列表拼接成张量
    spk_ref_gen_tensors = {}
    for spk, emb_data in spk_dict.items():
        if len(emb_data["ref_embs"]) == 0:
            continue
        ref_embs_tensor = torch.cat(emb_data["ref_embs"], dim=0)   # [N, 192]
        gen_embs_tensor = torch.cat(emb_data["gen_embs"], dim=0)   # [N, 192]
        spk_ref_gen_tensors[spk] = (ref_embs_tensor, gen_embs_tensor)
    
    return spk_ref_gen_tensors


def get_score(spk_ref_gen_tensors):
    spk_scores = {}
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for spk in spk_ref_gen_tensors.keys():
        ref_tensor, gen_tensor = spk_ref_gen_tensors[spk]
        scores = cos(ref_tensor, gen_tensor)
        aavg_score = torch.mean(scores).item()
        spk_scores[spk] = aavg_score
    
    if len(spk_scores) > 0:
        avg_score = sum(spk_scores.values()) / len(spk_scores)
    else:
        avg_score = 0.0
    
    return spk_scores, avg_score


# wav_dir = "../cosyvoice_0625/output/test300_v2/aa"
# spk2info_path = "trained_models/yoyo_sft/spk2info.pt"
# spk2info_path = "datas/yoyo_sft/train/spk2embedding.pt"


# wav_dir = "output/yoyo_sft_zz_epoch0_test300_rename/aa"
# spk2info_path = "trained_models/yoyo_sft/spk2info.pt"


spk2info_path = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/test2_spk2info.pt"  # 原始 spk emb
# spk2info_path = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/spk2info_enh_norm-23_trim.pt"  # enhance + norm -23 + trim
spk_emb_dict = get_spk_emb(spk2info_path)
print("1111111111111111: ", spk_emb_dict.keys())


# wav_dirs = ["others/out_spk_ref/minimax_out_audio",
#             "output/zero_shot_hindi/aa",
#             "output/zero_shot_hindi_enhance_norm/aa",
#             "output/zero_shot_hindi_yoyosft_raw/aa",
#             "output/zero_shot_hindi_yoyosft_enhance_norm/aa"]

# wav_dirs = ["output/zero_shot_hindi_yoyosft_enhance_norm-23/aa",
#             "output/zero_shot_hindi_yoyosft_enhance_norm-23_trim/aa"]
# wav_dirs = ["output2/zeroshot_bbc07230902_yoyo0904_thres480_avg3/aa",
#             "output2/zeroshot_bbc07230902_yoyo0904_thres480_avg5/aa"]

# wav_dirs = [
    # "others/out_spk_ref/minimax_out_audio",
    # "output5/yoyo_sft6spk_avg5",
    # "output5/yoyo_sft6spk_avg3",
    # "output5/bbc07230902_yoyo0904_thres300_avg5_zeroshot",
    # "output5/bbc07230902_yoyo0904_thres480_avg5_zeroshot",
    # "output5/bbc07230902_yoyo0904_thres600_avg5_zeroshot",
    # "output5/bbc07230902_yoyo0904_thres300_avg3_zeroshot",
    # "output5/bbc07230902_yoyo0904_thres480_avg3_zeroshot",
    # "output5/bbc07230902_yoyo0904_thres600_avg3_zeroshot",
    # "output5/yoyo_20250904_2spks_sft_basethres300avg5_epoch1",
    # "output5/yoyo_20250904_2spks_lora_basethres300avg5_epoch1",
    # "output5/yoyo_20250904_2spks_sft_baseyoyo6spkavg5_epoch1",
    # "output5/yoyo_20250904_2spks_lora_baseyoyo6spkavg5_epoch1",
    # ]

wav_dirs = [
    "output_test1min/test_1min_basethres300avg5_lora_epoch1",
    "output_test1min/test_1min_basethres300avg5_sft",
    "output_test1min/test_1min_baseyoyo6spkavg5_lora_epoch1",
    "output_test1min/test_1min_baseyoyo6spkavg5_sft",
]
    
    

for wav_dir in wav_dirs:
    spk_ref_gen_tensors = gen_pair(wav_dir, spk_emb_dict)
    spk_scores, avg_score = get_score(spk_ref_gen_tensors)
    print(f"{wav_dir} each spk score: {spk_scores}")
    print(f"{wav_dir} Average Cosine Similarity Score: {avg_score}")
