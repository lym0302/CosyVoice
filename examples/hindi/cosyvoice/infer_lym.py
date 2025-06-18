# coding = utf-8
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio
from tqdm import tqdm
import time
import os

model_dir = "/home/os/lym/cosyvoice/CosyVoice/pretrained_models/CosyVoice-300M_ft_hi"
cosyvoice = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=False)
# sft usage
print(cosyvoice.list_available_spks())
# change stream=True for chunk stream inference
# 目前还不能用 fronted， 对hindi的处理还有点问题
for i, j in enumerate(cosyvoice.inference_sft('ये जबरदस्ती हाँ क्यों? भिवा रेवा मुझसे।', '60145936', stream=False, text_frontend=False)):
# for i, j in enumerate(cosyvoice.inference_sft('hello, how are you', '60145936', stream=False, text_frontend=False)):
    torchaudio.save('warmup_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
print("warm up successfully.")


dur_all = 0.0
test_file = "../../../dataset/hindi/test.list"
outdir = "tts_hi"
os.makedirs(outdir, exist_ok=True)

st = time.time()
with open(test_file, 'r', encoding='utf-8') as fr:
    for line in tqdm(fr.readlines()):
        wav_path, spk, text, _ = line.strip().split("\t")
        utt = spk + "_" + wav_path.split("/")[-1].replace(".wav", "")
        # print("11111: ", utt, text)
        try:
            for i, j in enumerate(cosyvoice.inference_sft(text, spk, stream=False, text_frontend=False)):
                torchaudio.save(f'{outdir}/{utt}_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
                # print("222222222222: ", j['tts_speech'].shape)
                dur = j['tts_speech'].shape[1] / cosyvoice.sample_rate
                # print("dddddddddddddd: ", dur)
                dur_all += dur
        except Exception as e:
            print(f"eeeeeeeeeeeeerror on {utt}")

et = time.time()

spend_time = et - st
rtf = spend_time/dur_all
print(f"dur all: {dur_all}, spend time: {spend_time}, rtf: {rtf}")
