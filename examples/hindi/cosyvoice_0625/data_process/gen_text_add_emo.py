# coding=utf-8
import pandas as pd
from tqdm import tqdm
import argparse

emo2hindi = {
    "中立/neutral": "तटस्थ",       # 中立
    "开心/happy": "खुश",          # 开心
    "难过/sad": "दुखी",           # 难过
    "生气/angry": "गुस्सा",        # 生气
    "厌恶/disgusted": "घृणा",      # 厌恶
    "吃惊/surprised": "आश्चर्यचकित", # 吃惊
    "恐惧/fearful": "डर",          # 恐惧
    "<unk>": "मिश्रित भावनाएँ",    # 未知 → 混合情感
    "其他/other": "मिश्रित भावनाएँ" # 其他 → 混合情感
}


def read_wav_scp(wav_scp_file):
    utt2wav = {}
    with open(wav_scp_file) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
            
    return utt2wav


def load_emo_dict_and_stats(csv_path):
    """
    读取 CSV 文件，获取 {wav_path: emo} 字典，并统计 emo 占比

    参数:
        csv_path (str): CSV 文件路径，需包含表头 wav_path, emo, emo_score, emo_info

    返回:
        emo_dict (dict): key=wav_path, value=emo
        emo_stats (dict): key=emo类别, value=占比(百分比)
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 构造 wav_path → emo 的字典
    emo_dict = dict(zip(df["wav_path"], df["emo"]))

    # 统计 emo 分布
    emo_counts = df["emo"].value_counts(normalize=True) * 100  # 转换成百分比
    emo_stats = emo_counts.to_dict()
    
    for emo, pct in emo_stats.items():
        print(f"{emo}: {pct:.2f}%")

    return emo_dict


def generate_new_text_file(text_file, output_file, emo_dict, utt2wav):
    """
    生成新的 text_file，格式为：
    utt_id  印地语emo <|endofprompt|> 原始文本

    参数:
        text_file (str): 原始 text 文件路径
        output_file (str): 新生成的 text 文件路径
        emo_dict (dict): {wav_path: emo} 字典
        emo2hindi (dict): {emo: emo的印地语翻译}
    """
    with open(text_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin.readlines()):
            line = line.strip().replace('\n', '').split()
            if not line:
                continue
            utt = line[0]
            text = " ".join(line[1:])
            
            wav_path = utt2wav[utt]
            emo = emo_dict[wav_path]
            emo_hindi = emo2hindi.get(emo, "मिश्रित भावनाएँ")  # 默认混合情感

            new_text = f"{utt} {emo_hindi} <|endofprompt|> {text}"
            fout.write(new_text + "\n")


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成带印地语情感的 text 文件")
    parser.add_argument("--emo_csv_file", type=str, required=True, help="情感 CSV 文件路径")
    parser.add_argument("--wav_scp_file", type=str, required=True, help="wav.scp")
    parser.add_argument("--raw_text_file", type=str, required=True, help="原始 text 文件路径")
    parser.add_argument("--new_text_file", type=str, required=True, help="生成的新 text 文件路径")
    args = parser.parse_args()
    
    emo_csv_file = args.emo_csv_file
    wav_scp_file = args.wav_scp_file
    raw_text_file = args.raw_text_file
    new_text_file = args.new_text_file
    
    
    utt2wav = read_wav_scp(wav_scp_file)
    emo_dict = load_emo_dict_and_stats(emo_csv_file)
    generate_new_text_file(raw_text_file, new_text_file, emo_dict, utt2wav)


    
