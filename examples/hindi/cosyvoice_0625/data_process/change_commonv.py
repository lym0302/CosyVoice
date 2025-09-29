# coding=utf-8
from tqdm import tqdm

infile = "filelists_temp/filelist_commonvoice/data.list"
outfile = "filelists_temp/filelist_commonvoice/data_new.list"

with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
    for line in tqdm(fr.readlines()):
        line_list = line.strip().split("\t")
        if len(line_list) < 2:
            print(f"eeeeeeeeeeeeeerror in {line}")
            continue
        
        wav_path, spk = line_list[0], line_list[1]
        wav_path_list = wav_path.split("/")
        wav_dir = "/".join(wav_path_list[:-1])
        wavname = wav_path_list[-1]
        new_wavpath = f"{wav_dir}/{spk}/{wavname}"
        
        line_list[0] = new_wavpath
        
        fw.write("\t".join(line_list) + "\n")
        
