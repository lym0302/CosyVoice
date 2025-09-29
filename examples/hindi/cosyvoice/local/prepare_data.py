import argparse
import logging
import glob
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    with open(args.src_dir, 'r', encoding='utf-8') as fr:
        for line in tqdm(fr.readlines()):
            line_list = line.strip().split("\t")
            # if len(line_list) != 4:
            if len(line_list) != 5:
                print(line_list)
                continue
            # wav, spk, content, asr_conf = line_list
            wav, spk, content, dur, asr_conf = line_list
            
            utt = spk + "_" + os.path.basename(wav).replace('.wav', '')
            spk = utt.split('_')[0]
            utt2wav[utt] = wav
            utt2text[utt] = content
            utt2spk[utt] = spk
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',  # 这里变成我们之前生成的 data.list 文件
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()
