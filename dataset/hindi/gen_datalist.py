#codind=utf-8

import os
import json

audio_root = 'audio_data_asr_1000/audio_data_hi'
asr_root = 'audio_data_asr_1000/audio_data_hi_asr'
output_file = 'data.list'

with open(output_file, 'w', encoding='utf-8') as fout:
    for spk in os.listdir(audio_root):
        spk_audio_dir = os.path.join(audio_root, spk)
        spk_asr_dir = os.path.join(asr_root, spk)

        if not os.path.isdir(spk_audio_dir) or not os.path.isdir(spk_asr_dir):
            continue

        for wav_file in os.listdir(spk_audio_dir):
            if not wav_file.endswith('.wav'):
                continue

            wav_path = os.path.abspath(os.path.join(spk_audio_dir, wav_file))
            base_name = os.path.splitext(wav_file)[0]
            asr_txt_path = os.path.join(spk_asr_dir, base_name + '.txt')

            if not os.path.exists(asr_txt_path):
                print(f"[Warning] Missing ASR file: {asr_txt_path}")
                continue

            try:
                if os.path.getsize(asr_txt_path) == 0:
                    print(f"[Warning] Empty ASR file: {asr_txt_path}")
                    continue

                with open(asr_txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"[Warning] Blank ASR file: {asr_txt_path}")
                        continue

                    asr_data = json.loads(content)

                    text = asr_data.get("DisplayText", "").strip()
                    confidence = asr_data.get("NBest", [{}])[0].get("Confidence", 0.0)

                fout.write(f"{wav_path}\t{spk}\t{text}\t{confidence:.4f}\n")

            except json.JSONDecodeError as je:
                print(f"[Error] JSON parsing failed: {asr_txt_path} - {je}")
            except Exception as e:
                print(f"[Error] Failed to process {asr_txt_path}: {e}")
