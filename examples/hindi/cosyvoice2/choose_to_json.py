import json
import random

def choose(input_list, output_json, choose_num=200):
    # 读取全部数据
    with open(input_list, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) > choose_num:
        selected_lines = random.sample(lines, choose_num)
    else:
        selected_lines = lines

    # 构建输出字典
    result = {}
    for line in selected_lines:
        wav_path, spk, text, dur, asr_conf = line.strip().split("\t")
        utt = spk + "_" + wav_path.split("/")[-1].replace(".wav", "")
        result[utt] = [text]

    # 保存为 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(selected_lines)} 条数据到 {output_json}")


# choose("filelists/bbc07230902_yoyo0904/thres480/test.list", 
#        "datas/bbc07230902_yoyo0904_thres480/test30.json",
#        choose_num=30)
# exit()


def merge_json_files(path1, path2, output_path, keep='last'):
    """
    合并两个 JSON 文件并去重（按键），结果保存到 output_path。

    参数:
        path1 (str): 第一个 JSON 文件路径。
        path2 (str): 第二个 JSON 文件路径。
        output_path (str): 合并结果保存路径。
        keep (str): 'last' 表示 path2 的值覆盖 path1，'first' 表示保留 path1 的值。
    """
    with open(path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    if keep == 'last':
        merged = {**data1, **data2}
    elif keep == 'first':
        merged = {**data2, **data1}
    else:
        raise ValueError("参数 keep 应为 'first' 或 'last'")
    
    print("llllllllllllllllllllllll: ", len(merged))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"已合并 {path1} 和 {path2}，保存至 {output_path}")

# input_list = "filelists/test.list"
# output_json = "test200.json"
# choose(input_list, output_json)
# merge_json_files("test200.json", "test80.json", "test280.json")



import random
import json

def choose_to_jsonl(input_lists, output_json, choose_num=200):
    """
    从 input_list 中随机选择若干条音频信息，保存为 JSONL 文件
    
    参数:
        input_list (str): 输入文件，每行格式 audio_path \t spk \t text \t dur \t asr_conf
        output_json (str): 输出 JSONL 文件路径
        choose_num (int): 随机选择数量，默认 200
    """
    # 每个文件的 {audio_path: line} 映射
    file_dicts = []

    for input_file in input_lists:
        d = {}
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                audio_path = parts[0]
                d[audio_path] = line.strip()
        file_dicts.append(d)

    # 取所有文件 audio_path 的交集
    common_keys = set(file_dicts[0].keys())
    for d in file_dicts[1:]:
        common_keys &= set(d.keys())

    print(f"共有 {len(common_keys)} 条音频在所有文件中都出现")

    # 从交集里取出对应的行（以第一个文件为准）
    common_lines = [file_dicts[0][k] for k in common_keys]

    # 随机选择
    if len(common_keys) < 50 or len(common_lines) < choose_num:
        selected_lines = common_lines
    else:    
        selected_lines = random.sample(common_lines, choose_num)


    # 保存为 JSONL
    with open(output_json, "w", encoding="utf-8") as fout:
        for line in selected_lines:
            line = line.strip()
            if not line:
                continue
            try:
                wav_path, spk, text, dur, asr_conf = line.split("\t")
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue

            utt = wav_path.split("/")[-1].replace(".wav", "")
            item = {
                "utt": utt,
                "spk": spk,
                "text": text
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已生成 JSONL 文件: {output_json}, 共 {len(selected_lines)} 条记录")

        
# input_lists = ["filelists/bbc07230902_yoyo0904_thres300/test.list",
#                "filelists/bbc07230902_yoyo0904_thres480/test.list",
#                "filelists/bbc07230902_yoyo0904_thres600/test.list"]
# output_json = "filelists/bbc07230902_yoyo0904/test30.jsonl"
# choose_to_jsonl(input_lists, output_json, choose_num=30)


import pandas as pd

def load_emo_dict_and_stats(csv_path):
    """
    读取 CSV 文件，获取 {utt: emo} 字典，并统计 emo 占比

    参数:
        csv_path (str): CSV 文件路径，需包含表头 wav_path, emo, emo_score, emo_info

    返回:
        emo_dict (dict): key=utt, value=emo
        emo_stats (dict): key=emo类别, value=占比(百分比)
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 构造 utt → emo 的字典
    utt_list = []
    for wav_path in df["wav_path"]:
        spk = wav_path.split("/")[-2]
        fname = wav_path.split("/")[-1].replace(".wav", "")
        utt = f"{spk}_{fname}"
        utt_list.append(utt)

    emo_dict = dict(zip(utt_list, df["emo"]))

    # 统计 emo 分布
    emo_counts = df["emo"].value_counts(normalize=True) * 100  # 转换成百分比
    emo_stats = emo_counts.to_dict()
    
    for emo, pct in emo_stats.items():
        print(f"{emo}: {pct:.2f}%")

    return emo_dict


import json
from tqdm import tqdm
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

def add_emo(test_file, emo_csv_file, outfile):
    emo_dict = load_emo_dict_and_stats(emo_csv_file)

    with open(test_file, "r", encoding="utf-8") as fr, \
         open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr, desc=f"Processing {test_file}"):
            record = json.loads(line)
            utt = record["utt"]
            spk = record["spk"]
            text = record["text"]  # 注意这里是原始文本，不是 spk
            utt_key = f"{spk}_{utt}"

            if utt_key in emo_dict:
                emo = emo_dict[utt_key]
            else:
                emo = "<unk>"  # 如果没找到 emo，用未知
            
            emo_hindi = emo2hindi.get(emo, "मिश्रित भावनाएँ")

            # new_text = f"{emo_hindi} <|endofprompt|> {text}"

            out_record = {
                "utt": utt,
                "spk": spk,
                "text": text,
                "emo_hindi": emo_hindi,
                "emo": emo
            }
            fw.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"\n✅ 已生成带 emotion 的文件: {outfile}")

            
test_file = "datas/bbc07230902_yoyo0904_thres480/test30.jsonl"
emo_csv_file = "filelists/bbc07230902_yoyo0904_thres300/emo.csv"
outfile = "datas/bbc07230902_yoyo0904_thres480_emo/test30_emo.jsonl"
add_emo(test_file, emo_csv_file, outfile)