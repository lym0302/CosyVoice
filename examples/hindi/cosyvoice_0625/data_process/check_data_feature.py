import os
import torch
import argparse
import shutil



def read_utt_set_from_text(path):
    utt_set = set()
    with open(path, 'r') as f:
        for line in f:
            utt = line.strip().split()[0]
            utt_set.add(utt)
    return utt_set

def read_utt_set_from_spk2utt(path):
    utt_set = set()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt_set.update(parts[1:])
    return utt_set

def read_utt_set_from_pt(path):
    data = torch.load(path)
    return set(data.keys())

def save_spk2utt(path, utt_set, old_path):
    with open(path, 'w') as fout, open(old_path, 'r') as fin:
        for line in fin:
            parts = line.strip().split()
            spk, utts = parts[0], parts[1:]
            new_utts = [utt for utt in utts if utt in utt_set]
            if new_utts:
                fout.write(f"{spk} {' '.join(new_utts)}\n")

def save_filtered_text_file(path, utt_set, old_path):
    with open(path, 'w') as fout, open(old_path, 'r') as fin:
        for line in fin:
            utt = line.strip().split()[0]
            if utt in utt_set:
                fout.write(line)

def save_filtered_pt_file(path, utt_set, old_path):
    data = torch.load(old_path)
    new_data = {utt: data[utt] for utt in utt_set if utt in data}
    torch.save(new_data, path)

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 文件路径
    text_path = os.path.join(input_dir, "text")
    utt2spk_path = os.path.join(input_dir, "utt2spk")
    wav_scp_path = os.path.join(input_dir, "wav.scp")
    spk2utt_path = os.path.join(input_dir, "spk2utt")
    utt2embedding_path = os.path.join(input_dir, "utt2embedding.pt")
    utt2speech_token_path = os.path.join(input_dir, "utt2speech_token.pt")
    spk2emb_path = os.path.join(input_dir, "spk2embedding.pt")

    # 收集所有文件中出现的 utt
    sets = [
        read_utt_set_from_text(text_path),
        read_utt_set_from_text(utt2spk_path),
        read_utt_set_from_text(wav_scp_path),
        read_utt_set_from_spk2utt(spk2utt_path),
        read_utt_set_from_pt(utt2embedding_path),
        read_utt_set_from_pt(utt2speech_token_path)
    ]

    common_utts = set.intersection(*sets)
    print(f"✅ 有效 utt 的交集数量为: {len(common_utts)}")

    # 覆盖写入新版本文件（只保留交集内 utt）
    new_text_path = os.path.join(output_dir, "text")
    new_utt2spk_path = os.path.join(output_dir, "utt2spk")
    new_wav_scp_path = os.path.join(output_dir, "wav.scp")
    new_spk2utt_path = os.path.join(output_dir, "spk2utt")
    new_utt2embedding_path = os.path.join(output_dir, "utt2embedding.pt")
    new_utt2speech_token_path = os.path.join(output_dir, "utt2speech_token.pt")
    new_spk2emb_path = os.path.join(output_dir, "spk2embedding.pt")
    
    save_spk2utt(new_spk2utt_path, common_utts, spk2utt_path)
    save_filtered_text_file(new_text_path, common_utts, text_path)
    save_filtered_text_file(new_utt2spk_path, common_utts, utt2spk_path)
    save_filtered_text_file(new_wav_scp_path, common_utts, wav_scp_path)
    save_filtered_pt_file(new_utt2embedding_path, common_utts, utt2embedding_path)
    save_filtered_pt_file(new_utt2speech_token_path, common_utts, utt2speech_token_path)
    shutil.copy(spk2emb_path, new_spk2emb_path)
    
    print(f"✅ 所有文件已清洗，仅保留交集 utt 到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗语音数据文件")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="输入文件夹路径")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="输出文件夹路径")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
