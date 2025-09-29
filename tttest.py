import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import time
from tqdm import tqdm
import os

# cosyvoice2 use vllm
# from vllm import ModelRegistry
# from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
# ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
from cosyvoice.cli.cosyvoice import CosyVoice2

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)


prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
## zero-shot
# text = "[laughter]有时候，看着小孩子们的天真行为[laughter]，我们总会会心一笑。"  # 可以实现
# text = "She pursued her dreams with <strong>enthusiasm</strong> and <strong>grit</strong>"  # 有一点
# text = "<laughter>有时候，看着小孩子们的天真行为</laughter>，我们总会会心一笑。"  # 有一点
text = "[breath]有时候，看着小孩子们的天真行为[breath]，我们总会会心一笑。"
for i, j in enumerate(cosyvoice.inference_zero_shot(text, "希望你以后能够做的比我还好呦。", prompt_speech_16k, stream=False)):
    torchaudio.save('zeroshot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
### instruct
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用伤心的情感说', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
exit()

# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False, text_frontend=False)):
#     torchaudio.save('aaa.wav', j['tts_speech'], cosyvoice.sample_rate)


def init_cosyv1(model_path, load_jit=True, load_trt=False, fp16=False):
    cosyvoice = CosyVoice(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16) # 会更快一点点
    sr = cosyvoice.sample_rate
    # sft usage
    print(cosyvoice.list_available_spks())
    # warm up
    for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
        torchaudio.save('tttemp.wav', j['tts_speech'], sr)
        print("1111111111111111: ", j['tts_speech'].shape, sr)
    return cosyvoice, sr

# init_cosyv1(model_path="pretrained_models/CosyVoice-300M-SFT")


def test_cosyvoice1(test_file="test.txt", out_dir="output_jit_true", ):
    # cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
    model_path = 'pretrained_models/CosyVoice-300M-SFT'
    cosyvoice, sr = init_cosyv1(model_path, load_jit=True) # warm up done
    os.makedirs(out_dir, exist_ok=True)
    
    count = 0
    dur = 0.0 
    st = time.time()
    with open(test_file, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            try:
                utt_id, text = line.strip().split("|")
                for i, j in enumerate(cosyvoice.inference_sft(text, '中文女', stream=False)):
                    audio_gen = j['tts_speech']
                    torchaudio.save(f'{out_dir}/{utt_id}_{i}.wav', audio_gen, sr)
                    dur += (audio_gen.shape[1] / sr)
                count += 1
            except Exception as e:
                print(f"eeeeeeeeeeeeeeeeeeerror in {line} on {e}")

    et = time.time()
    infer_time = et - st
    rtf = infer_time / dur
    print(f"Count: {count}, Infer Time: {infer_time}, Dur: {dur}, RTF: {rtf}")
    


def init_cosyv2(model_path, prompt_text, prompt_audio, spk_name = "spk_aaa", save_spk=True, 
                load_jit=False, load_trt=False, load_vllm=False, fp16=False):
    cosyvoice = CosyVoice2(model_path, load_jit=load_jit, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16)
    sr = cosyvoice.sample_rate
    zero_shot_spk_id = ''
    # warm up
    text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio, stream=False)):
        torchaudio.save('tttemp.wav'.format(i), j['tts_speech'], sr)
    if save_spk:
        assert cosyvoice.add_zero_shot_spk(prompt_text, prompt_audio, spk_name) is True
        zero_shot_spk_id = spk_name   
    return cosyvoice, sr, zero_shot_spk_id


def test_cosyvoice2(test_file="test.txt", out_dir="output/output_cosyv2"):
    model_path = 'pretrained_models/CosyVoice2-0.5B'
    prompt_text = "希望你以后能够做的比我还好呦。"
    prompt_audio = load_wav('./asset/zero_shot_prompt.wav', 16000) 
    cosyvoice, sr, zero_shot_spk_id = init_cosyv2(model_path, prompt_text, prompt_audio) # warm up done
    print("aaaaaaaaaaaaaaa: ", zero_shot_spk_id)
    os.makedirs(out_dir, exist_ok=True)
    
    count = 0
    dur = 0.0 
    st = time.time()
    with open(test_file, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()[:2]):
            try:
                utt_id, text = line.strip().split("|")
                  
                for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio, zero_shot_spk_id=zero_shot_spk_id, stream=False)):
                    audio_gen = j['tts_speech']
                    torchaudio.save(f'{out_dir}/{utt_id}_{i}.wav', audio_gen, sr)
                    dur += (audio_gen.shape[1] / sr)
                count += 1
                
            except Exception as e:
                print(f"eeeeeeeeeeeeeeeeeeerror in {line} on {e}")

    et = time.time()
    infer_time = et - st
    rtf = infer_time / dur
    print(f"Count: {count}, Infer Time: {infer_time}, Dur: {dur}, RTF: {rtf}")


# test_cosyvoice2()
            
