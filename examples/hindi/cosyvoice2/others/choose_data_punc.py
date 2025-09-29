# coding=utf-8

import random
from tqdm import tqdm
import os
import shutil
from urllib.request import urlopen


def choose(infile, outfile, num_per_class=700):
    """
    根据 text 末尾标点选择印地语数据：
    - 竖线 |
    - 英文问号 ?
    - 没有标点（Devanagari 字母/数字/空格）
    各选 num_per_class 条，保存到 outfile
    """

    def is_devanagari_char(c):
        # Devanagari 字母 \u0900-\u097F，数字 \u0966-\u096F，空格
        return ("\u0900" <= c <= "\u097F") or ("\u0966" <= c <= "\u096F") or c == " "

    class_pipe = []
    class_question = []
    class_none = []

    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue  # 略过格式不对的行
            wav_path, spk, text, dur, asr_conf = parts
            text = text.strip()
            if not text:
                continue
            last_char = text[-1]
            if last_char == "।":
                class_pipe.append(line)
            elif last_char == "?":
                class_question.append(line)
            elif is_devanagari_char(last_char):
                class_none.append(line)
            else:
                continue  # 其他标点忽略

    # 打印每类数量
    print(f"竖线 | 类: {len(class_pipe)} 条")
    print(f"英文问号 ? 类: {len(class_question)} 条")
    print(f"无标点类: {len(class_none)} 条")

    # 随机抽取 num_per_class 条
    def sample_list(lst):
        if len(lst) >= num_per_class:
            return random.sample(lst, num_per_class)
        else:
            print(f"警告：该类数量不足 {num_per_class}，只取 {len(lst)} 条")
            return lst

    selected_lines = sample_list(class_pipe) + sample_list(class_question) + sample_list(class_none)

    # 打乱顺序
    random.shuffle(selected_lines)

    # 写入输出文件
    with open(outfile, "w", encoding="utf-8") as f:
        for line in selected_lines:
            f.write(line)

    print(f"已将选中的数据保存到 {outfile}")


infile = "filelists_v3/yoyo_v2/data_hindi_asrconf_0.7.list"
outfile = "filelists_v3/yoyo_v2/data_hindi_asrconf_0.7_choose2100.list"
choose(infile, outfile, num_per_class=700)




def copy_or_download_selected_audio(list_file, target_dir):
    """
    将 data_selected.list 中的 wav_path 文件复制或下载到 target_dir
    """
    os.makedirs(target_dir, exist_ok=True)

    with open(list_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    for line in tqdm(lines):
        parts = line.strip().split("\t")
        if len(parts) < 1:
            continue
        wav_path = parts[0]
        filename = os.path.basename(wav_path)
        target_path = os.path.join(target_dir, filename)

        try:
            if wav_path.startswith("http"):
                # 下载文件
                with urlopen(wav_path) as response, open(target_path, "wb") as out_file:
                    out_file.write(response.read())
            else:
                # 本地文件直接复制
                if not os.path.isfile(wav_path):
                    print(f"警告：文件不存在 {wav_path}")
                    continue
                shutil.copy(wav_path, target_path)
            count += 1
        except Exception as e:
            print(f"错误：处理 {wav_path} 时失败，原因: {e}")

    print(f"已处理 {count} 个音频文件到 {target_dir}")


copy_or_download_selected_audio("filelists_v3/yoyo_v2/data_hindi_asrconf_0.7_choose2100.list", 
                                "filelists_v3/yoyo_v2/selected2100_audios")
