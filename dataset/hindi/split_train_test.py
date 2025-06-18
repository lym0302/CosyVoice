import random
from collections import defaultdict

random.seed(42)  # 固定随机种子，保证结果可复现

input_list = 'data.list'
train_file = 'train.list'
dev_file = 'dev.list'
test_file = 'test.list'

# Step 1: 按说话人分组
spk_data = defaultdict(list)

with open(input_list, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue  # 跳过异常行
        spk = parts[1]
        spk_data[spk].append(line)

# Step 2: 随机分配每个说话人样本到 dev / test / train
train_lines, dev_lines, test_lines = [], [], []

for spk, lines in spk_data.items():
    total = len(lines)
    if total < 100:
        print(f"[Warning] {spk} has only {total} samples, skipping")
        continue

    random.shuffle(lines)
    selected_dev = lines[:50]
    selected_test = lines[50:100]
    selected_train = lines[100:]

    dev_lines.extend(selected_dev)
    test_lines.extend(selected_test)
    train_lines.extend(selected_train)

# Step 3: 写入文件
with open(dev_file, 'w', encoding='utf-8') as f:
    f.writelines(dev_lines)

with open(test_file, 'w', encoding='utf-8') as f:
    f.writelines(test_lines)

with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

print(f"[✓] 完成划分: train({len(train_lines)}), dev({len(dev_lines)}), test({len(test_lines)})")
