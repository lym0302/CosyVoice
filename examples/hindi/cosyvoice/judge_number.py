# coding=utf-8
from tqdm import tqdm

def judge(infile, outfile):
    count = 0
    count_left = 0
    with open(infile, 'r', encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            text = line_list[2]
            count += 1
            if not any(c.isdigit() for c in text):
                fw.write(line)
                count_left += 1
    return count, count_left

# data_dir="/data/liangyunming/tts_20250618/CosyVoice/dataset/hindi/20250618"
# for mode in ['train', 'dev', 'test']:
#     infile = f"{data_dir}/{mode}.list"
#     count, count_left = judge(infile)
#     print("cccccccccccccccc: ", mode, count, count_left)

infile = "/data/liangyunming/tts_20250618/CosyVoice/dataset/hindi/20250618_norm/data.list"
outfile = "/data/liangyunming/tts_20250618/CosyVoice/dataset/hindi/20250618_norm_rmnum/data.list"
count, count_left = judge(infile, outfile)
print("cccccccccccccccc: ", count, count_left)
            