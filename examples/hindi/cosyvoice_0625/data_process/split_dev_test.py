# coding=utf-8
import argparse
import random
from collections import defaultdict
from math import floor

def split_data(input_file: str, train_file: str, dev_file: str, test_file: str, dev_ratio=0.05, test_ratio=0.05, seed=42, max_limit=100, count_thres=100):
    random.seed(seed)

    spk2lines = defaultdict(list)
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            wav_path, spk, text, dur, asr_conf = parts
            spk2lines[spk].append(line)

    train_lines = []
    dev_lines = []
    test_lines = []
    
    for spk, lines in spk2lines.items():
        n = len(lines)
        if n > count_thres:
            n_dev = min(floor(n * dev_ratio), max_limit)
            n_test = min(floor(n * test_ratio), max_limit)
            n_train = n - n_dev - n_test

            random.shuffle(lines)

            dev_lines.extend(lines[:n_dev])
            test_lines.extend(lines[n_dev:n_dev + n_test])
            train_lines.extend(lines[n_dev + n_test:])
        else:
            # 小于等于阈值的全放 train
            train_lines.extend(lines)

    with open(train_file, "w", encoding="utf-8") as ftrain, \
         open(dev_file, "w", encoding="utf-8") as fdev, \
         open(test_file, "w", encoding="utf-8") as ftest:
        ftrain.writelines(train_lines)
        fdev.writelines(dev_lines)
        ftest.writelines(test_lines)

    print(f"Split done: train={len(train_lines)}, dev={len(dev_lines)}, test={len(test_lines)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data.list into train/dev/test by speaker")
    parser.add_argument("-i", "--input_file", required=True, help="输入的 data.list 文件路径")
    parser.add_argument("-train", "--train_file", required=True, help="输出的 train.list 文件路径")
    parser.add_argument("-dev", "--dev_file", required=True, help="输出的 dev.list 文件路径")
    parser.add_argument("-test", "--test_file", required=True, help="输出的 test.list 文件路径")
    parser.add_argument("--dev_ratio", type=float, default=0.05, help="每个说话人用于dev的比例，默认0.05")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="每个说话人用于test的比例，默认0.05")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    parser.add_argument("--max_limit", type=int, default=100, help="每个说话人最大数量的 dev 和test")
    parser.add_argument("--count_thres", type=int, default=100, help="每个说话人数量大于这个值才会分 dev 和 test")

    args = parser.parse_args()
    split_data(args.input_file, args.train_file, args.dev_file, args.test_file,
               dev_ratio=args.dev_ratio, test_ratio=args.test_ratio, seed=args.seed, max_limit=args.max_limit, count_thres=args.count_thres)
