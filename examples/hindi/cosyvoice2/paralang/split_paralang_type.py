# coding=utf-8
import csv
import re
import os
from collections import defaultdict
from tqdm import tqdm


def split_tag(infile):
    """
    解析 infile 的第四列（无表头），提取副语言标签，返回 {tag: [lines...]} 的字典
    """
    tag_dict = defaultdict(list)

    with open(infile, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            if len(row) < 4:
                continue
            text = row[3].strip()  # 第四列

            tags = set()

            # 1. emotion=xxx
            # emo_match = re.findall(r"emotion=([a-zA-Z]+)", text)
            # for emo in emo_match:
            #     tags.add(f"emotion={emo}")

            # 2. <xxx>...</xxx>，只取标签名
            inline_tags = re.findall(r"<(?!/)(.*?)>", text)
            for tag in inline_tags:
                tags.add(f"<{tag}>")

            # 3. [xxx], 这里也会判断 emotion
            bracket_tags = re.findall(r"\[(.*?)\]", text)
            for tag in bracket_tags:
                tags.add(f"[{tag}]")

            # 将当前行加入对应标签
            for tag in tags:
                tag_dict[tag].append(row)

    return tag_dict


def write_to_csv(tag_dict, outdir):
    """
    将 tag_dict 中的内容写入多个 CSV 文件（无表头）
    """
    os.makedirs(outdir, exist_ok=True)
    print(f"There is {len(tag_dict)} tag types.")

    for tag, lines in tqdm(tag_dict.items()):
        # 文件名处理：去掉特殊符号，前缀加 "tag_"
        # safe_tag = tag.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
        safe_tag = tag
        filename = f"tag_{safe_tag}.csv"
        filepath = os.path.join(outdir, filename)

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(lines)

        print(f"✅ 已写入 {filepath}, 共 {len(lines)} 行")


def choose_tar_tag(infile, outfile, tar_tags=["laughter", "breath", "cough", "strong"]):
    with open(infile, "r", encoding="utf-8") as f, open(outfile, "w", encoding="utf-8", newline="") as out_f:
        reader = csv.reader(f)
        writer = csv.writer(out_f)
        for row in tqdm(reader):
            if len(row) < 4:
                continue
            text = row[3].strip()  # 第四列
            # 检查是否包含任意一个 tar_tags
            if any(tag in text for tag in tar_tags):
                writer.writerow(row)
                
def change(infile, outfile, audio_dir="/data/liuchao/annotation/audio"):
    with open(infile, "r", encoding="utf-8") as f, open(outfile, "w", encoding="utf-8", newline="") as out_f:
        reader = csv.reader(f)
        writer = csv.writer(out_f)
        # 写表头
        writer.writerow(["org_wav_path", "wav_path", "org_text", "org_text_paralang", "text"])
        
        for row in tqdm(reader):
            if len(row) < 4:
                continue
            org_wav_path, wav_path, org_text, org_text_paralang = row
            
            dst_audio_path = wav_path.strip().replace("http://parrot.com:8081/annotation/audio", audio_dir)
            text = re.sub(r'^\[emotion=[^\]]+\]\s*', '', org_text_paralang.strip())
            
            if os.path.exists(dst_audio_path):
                new_row = [org_wav_path, dst_audio_path, org_text, org_text_paralang, text]
                writer.writerow(new_row)
            else:
                print(f"Cannot find {dst_audio_path}")

change("results/choose_laughter_breath_cough_strong.csv",
       "to_eval/playlist.csv")
exit()
                
    


def main(infile, outdir):
    tag_dict = split_tag(infile)
    write_to_csv(tag_dict, outdir)
    # 按行数从大到小排序
    sorted_tags = sorted(tag_dict.items(), key=lambda x: len(x[1]), reverse=True)
    for tag, lines in sorted_tags:
        print(tag, len(lines))


if __name__ == "__main__":
    infile = "results/data_hindi_asr_merged_annotation.csv"   # 输入文件（无表头）
    outdir = "results/split_tag" # 输出目录
    # main(infile, outdir)
    outfile = "results/choose_laughter_breath_cough_strong.csv"
    # choose_tar_tag(infile, outfile)
