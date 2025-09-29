import os
from pydub import AudioSegment
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def process_one_line(line, raw_audio_dir, new_audio_dir):
    try:
        line_list = line.strip().split("\t")
        if len(line_list) < 3:
            return None
        wavname = line_list[0]
        spk = line_list[1]
        text = " ".join(line_list[2:]).strip()
        # wavname, spk, text = line.strip().split("\t")
        mp3_path = os.path.join(raw_audio_dir, wavname)
        spk_dir = os.path.join(new_audio_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)

        new_wavname = os.path.splitext(wavname)[0].replace(" ", "_") + ".wav"
        out_wav_path = os.path.join(spk_dir, new_wavname)

        # audio = AudioSegment.from_file(mp3_path).set_channels(1)
        # audio.export(out_wav_path, format="wav")
        if not os.path.exists(out_wav_path):
            audio = AudioSegment.from_file(mp3_path).set_channels(1)
            audio.export(out_wav_path, format="wav")

        waveform, sr = torchaudio.load(out_wav_path)
        duration = waveform.shape[1] / sr

        return f"{out_wav_path}\t{spk}\t{text}\t{duration:.3f}\t1.0\n"
    except Exception as e:
        print(f"Error processing {line.strip()}: {e}")
        return None

def main(args):
    with open(args.label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    os.makedirs(args.new_audio_dir, exist_ok=True)
    results = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_one_line, line, args.raw_audio_dir, args.new_audio_dir) for line in lines]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            res = future.result()
            if res:
                results.append(res)

    with open(args.output_list, "w", encoding="utf-8") as f:
        f.writelines(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert mp3 to wav and generate data.list")
    parser.add_argument("-i", "--raw_audio_dir", type=str, required=True, help="Path to raw audio directory")
    parser.add_argument("-o", "--new_audio_dir", type=str, required=True, help="Path to save converted wav files")
    parser.add_argument("-l", "--label_file", type=str, required=True, help="Path to label file")
    parser.add_argument("--output_list", type=str, default="/data/liangyunming/dataset/hindi/ttspart2/data.list", help="Output list file path")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()
    main(args)

    