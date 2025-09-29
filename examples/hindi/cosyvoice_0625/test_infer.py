import sys
sys.path.append('../../../third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio
import time
from tqdm import tqdm
import os
import torch
import random
import json

def gen_spk2info(infile, outfile):
    # 读取原始文件（包含list作为value）
    spk2emb = torch.load(infile, map_location="cpu")
    # 将所有list转为tensor
    spk2info = {
        spk: {"embedding": torch.tensor(emb).unsqueeze(0)}  # 增加一个 batch 维度
        for spk, emb in spk2emb.items()
    }
    torch.save(spk2info, outfile)
    # dict_keys(['65361586', '33798415', '79443944', '30629019', '49896910', '65427504'])
    print(f"转换完成，已保存为 {outfile}")

# gen_spk2info(infile="data_yoyo_sft/train/spk2embedding.pt", 
#              outfile="trained_models/yoyo_sft_basebbc240_epoch17/spk2info.pt")

model_path = 'trained_models/yoyo_sft_basebbc240_epoch17'

def init_cosyv1(model_path, load_jit=False, load_trt=False, fp16=False):
    cosyvoice = CosyVoice(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16) # 会更快一点点
    sr = cosyvoice.sample_rate
    # sft usage
    support_spks = cosyvoice.list_available_spks()
    print("spk list: ", support_spks) #['65361586', '33798415', '79443944', '30629019', '49896910', '65427504']
    # warm up
    for i, j in enumerate(cosyvoice.inference_sft('ठीक है कि कुछ उनके बीच में कुछ ऐसे हो जाता है कि औकात पर बात आ जाती है।', '65361586', stream=False, text_frontend=False)):
        torchaudio.save('tttemp.wav', j['tts_speech'], sr)
        print("1111111111111111: ", j['tts_speech'].shape, sr)
    return cosyvoice, sr, support_spks
    
   


def test_cosyvoice1(test_file="test_hindi_300.jsonl", out_dir="output/test300_v2"):
    cosyvoice, sr, support_spks = init_cosyv1(model_path)  # 初始化模型和支持的说话人
    os.makedirs(out_dir, exist_ok=True)

    # 读取测试文本
    with open(test_file, "r", encoding="utf-8") as fr:
        data = [json.loads(line.strip()) for line in fr if line.strip()]
    
    # 根据说话人数量平均分配句子
    total_sentences = len(data)
    num_spks = len(support_spks)
    per_spk_count = total_sentences // num_spks

    # 随机打乱语句
    random.seed(42)        # 设置随机种子
    random.shuffle(data)   # 打乱顺序

    # 分配句子
    spk2samples = {}
    for i, spk in enumerate(support_spks):
        start_idx = i * per_spk_count
        end_idx = (i + 1) * per_spk_count if i < num_spks - 1 else total_sentences
        spk2samples[spk] = data[start_idx:end_idx]
        
    # from tqdm import tqdm

    # output_jsonl = "test300.jsonl"
    # with open(output_jsonl, "w", encoding="utf-8") as fw:
    #     for spk, samples in tqdm(spk2samples.items(), desc="Saving JSONL"):
    #         for sample in samples:
    #             new_sample = dict(sample)
    #             new_sample["spk"] = spk
    #             fw.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

    count = 0
    dur = 0.0
    st = time.time()

    for spk, samples in spk2samples.items():
        for sample in tqdm(samples, desc=f"Processing {spk}"):
            try:
                utt_id = sample["utt"]
                text = sample["text"]
                for i, j in enumerate(cosyvoice.inference_sft(text, spk, stream=False, text_frontend=False)):
                    audio_gen = j['tts_speech']
                    torchaudio.save(f'{out_dir}/{utt_id}_{spk}_{i}.wav', audio_gen, sr)
                    dur += (audio_gen.shape[1] / sr)
                count += 1
            except Exception as e:
                print(f"❌ Error processing {sample} for speaker {spk}: {e}")

    et = time.time()
    infer_time = et - st
    rtf = infer_time / dur
    print(f"✅ Done. Count: {count}, Infer Time: {infer_time:.2f}s, Total Dur: {dur:.2f}s, RTF: {rtf:.4f}")
    
    
test_cosyvoice1()