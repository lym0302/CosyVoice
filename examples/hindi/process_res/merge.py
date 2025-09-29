import pandas as pd
import csv
import re
import jiwer
import math
# # from jiwer import RemovePunctuation, ToLowerCase, RemoveMultipleSpaces, RemoveEmptyStrings
# import jiwer.transforms as tr

def read_xlsx(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)  # 返回所有子表格
    sheets_dict = {}
    sheet_columns = {}

    for sheet_name, df in sheets.items():
        # 每行作为一个 dict 存储
        sheets_dict[sheet_name] = df.to_dict(orient='records')
        
        # 获取列名列表
        sheet_columns[sheet_name] = df.columns.tolist()

    return sheets_dict, sheet_columns


def get_new_dict(dict_list, tar_key):
    new_dict = {}
    for i in range(len(dict_list)):
        raw_dict = dict_list[i]
        new_key = raw_dict[tar_key].replace(".wav", "")
        new_dict[new_key] = {}
        for k in raw_dict.keys():
            if k != tar_key:
                new_dict[new_key][k] = raw_dict[k]
    return new_dict
        

def get_wer(raw_dict, gen_dict):
    def remove_hindi_punctuation(text: str) -> str:
        # 去除指定标点符号
        # text = re.sub(r"[।॥!?.,\"'“”‘’]", "", text)
        text = re.sub(r"[।॥|｜!?.,\"'“”‘’]", "", text)  # 去掉长短竖线
        # 将多个连续空格替换成一个空格
        text = re.sub(r"\s+", " ", text)
        # 去除首尾空格
        text = text.strip()
        return text
    
    common_keys = list(set(raw_dict.keys()) & set(gen_dict.keys()))
    raw_texts = []
    gen_texts = []
    raw_texts_haveprounc = []
    gen_texts_haveprounc = []
    for utt in common_keys:
        gen_text = gen_dict[utt]
        # 判定非法文本的条件
        if gen_text is None or \
           isinstance(gen_text, float) and math.isnan(gen_text) or \
           str(gen_text).strip().lower() in ["na", "nan", "not hindi"]:
            continue
        raw_text = raw_dict[utt]
        raw_texts.append(remove_hindi_punctuation(raw_text))
        gen_texts.append(remove_hindi_punctuation(gen_text))
        raw_texts_haveprounc.append(raw_text)
        gen_texts_haveprounc.append(gen_text)
    
    print("rrrrrrrrrrrrrrrrr: ", raw_texts, len(raw_texts))
    print("ggggggggggggggggg: ", gen_texts, len(gen_texts))
    wer_value = jiwer.wer(raw_texts, gen_texts)
    wer_value_havapunc = jiwer.wer(raw_texts_haveprounc, gen_texts_haveprounc)
    
    return round(wer_value, 3), round(wer_value_havapunc, 3) 
    

def write_merge(sheets_dict, output_file):
    # {'score': ['utt', 'Acc(har)', 'Natural(har)', 'Acc(raj)', 'Natural(raj)', 'Acc(ansh)', 'Natural(ansh)'], 
    # 'text': ['utt', 'rajan', 'harneet', 'ansh'], 
    # 'swag': ['wav_path', 'text', 'accuracy', 'realness'], 
    # 'input': ['utt', 'text']}
    tar_key_dict = {"score": "utt", "text": "utt", "swag": "wav_path", "input": "utt"}
    new_dicts = {}
    for k in sheets_dict.keys():
        new_dicts[k] = get_new_dict(sheets_dict[k], tar_key_dict[k])
    
    lines = []
    text_inp = {}
    text_swag = {}
    text_har = {}
    text_raj = {}
    text_ansh = {}
    for utt in new_dicts['swag'].keys():
        text_inp[utt] = new_dicts['input'][utt.replace("_18", "")]['text']
        text_swag[utt] = new_dicts['swag'][utt]['text']
        text_har[utt] = new_dicts['text'][utt]['harneet']
        text_raj[utt] = new_dicts['text'][utt]['rajan']
        text_ansh[utt] = new_dicts['text'][utt]['ansh']
        line = [utt, new_dicts['input'][utt.replace("_18", "")]['text'], \
                new_dicts['swag'][utt]['text'], new_dicts['swag'][utt]['accuracy'], new_dicts['swag'][utt]['realness'], \
                new_dicts['text'][utt]['harneet'], new_dicts['score'][utt]['Acc(har)'],  new_dicts['score'][utt]['Natural(har)'], \
                new_dicts['text'][utt]['rajan'], new_dicts['score'][utt]['Acc(raj)'],  new_dicts['score'][utt]['Natural(raj)'], \
                new_dicts['text'][utt]['ansh'], new_dicts['score'][utt]['Acc(ansh)'],  new_dicts['score'][utt]['Natural(ansh)']]
        
        lines.append(line)
    
    # 写入 CSV 文件
    header = [
        "utt", "text_inp", 
        "text_swag", "acc_swag", "nat_swag",
        "text_harneet", "acc_harneet", "nat_harneet",
        "text_rajan", "acc_rajan", "nat_rajan",
        "text_ansh", "acc_ansh", "nat_ansh",
    ]

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(lines)
    
    wer_dict = {}
    wer_dict['swag'] = get_wer(text_inp, text_swag)
    wer_dict['har'] = get_wer(text_inp, text_har)
    wer_dict['raj'] = get_wer(text_inp, text_raj)
    wer_dict['ansh'] = get_wer(text_inp, text_ansh)
    
    print("22222222222222222: ", wer_dict)
    

file_path = "test30_res_all.xlsx"
sheets_dict, sheet_columns = read_xlsx(file_path)
# print(sheets_dict)
print(sheet_columns)
output_file = "test30_res_merge.csv"
write_merge(sheets_dict, output_file)