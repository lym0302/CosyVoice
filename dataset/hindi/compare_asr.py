# coding=utf-8
from tqdm import tqdm



def get_res(file_path):
    res = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line_list = line.strip().split("\t")
            if ".wav" in line_list[0]:
                utt = line_list[1] + "_" + line_list[0].split("/")[-1].replace(".wav", "")
            else:
                utt = line_list[0]
            res[utt] = line_list[2]
    return res
            
def compare_dicts(dict1, dict2, output_file):
    # 获取两个 dict 中的共同 key
    common_keys = set(dict1.keys()) & set(dict2.keys())

    same_key_count = len(common_keys)
    with open(output_file, 'w', encoding='utf-8') as f:
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            f.write(f"{key}\t{val1}\t{val2}\t{dict1[key] == dict2[key]}\n")
            same_value_count = sum(1 for key in common_keys if dict1[key] == dict2[key])

    return same_key_count, same_value_count


infile1 = "raw_asr.list"
infile2 = "tts_hi_pretrained.list"
outfile = "compare_res.txt"
res1 = get_res(infile1)
res2 = get_res(infile2)
same_key_count, same_value_count = compare_dicts(res1, res2, outfile)
print(same_key_count, same_value_count)



