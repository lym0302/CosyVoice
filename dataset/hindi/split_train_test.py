import random
from collections import defaultdict

random.seed(42)  # 固定随机种子，保证结果可复现

root_dir = "20250618_norm_rmnum"
input_list = f'{root_dir}/data.list'
train_file = f'{root_dir}/train.list'
dev_file = f'{root_dir}/dev.list'
test_file = f'{root_dir}/test.list'
set_dev_thres = 1000   # 该说话人的条数大于这个值才会被分出一点dev 和 test
dev_each_spk = 20
mincount_each_spk = 20

spk_left = 0

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
    if total < mincount_each_spk:
        continue
    elif total > set_dev_thres:
        random.shuffle(lines)
        selected_dev = lines[:dev_each_spk]
        selected_test = lines[dev_each_spk:dev_each_spk*2]
        selected_train = lines[dev_each_spk*2:]
        dev_lines.extend(selected_dev)
        test_lines.extend(selected_test)
        train_lines.extend(selected_train)
        spk_left += 1
    else:
        random.shuffle(lines)
        train_lines.extend(lines)
        spk_left += 1
        

# Step 3: 写入文件
with open(dev_file, 'w', encoding='utf-8') as f:
    f.writelines(dev_lines)

with open(test_file, 'w', encoding='utf-8') as f:
    f.writelines(test_lines)

with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

print(f"[✓] 完成划分: train({len(train_lines)}), dev({len(dev_lines)}), test({len(test_lines)}), spk: {spk_left}")
