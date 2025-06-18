#coding=utf-8
import torch

# 加载原始 pt 文件
old_data = torch.load('data/train/spk2embedding.pt')  # 假设原文件叫 input.pt

# 新的格式化结果
spk2info = {}

for spkid, emb_list in old_data.items():
    emb_tensor = torch.tensor(emb_list).unsqueeze(0)  # 变成 shape [1, 192]
    spk2info[spkid] = {
        'embedding': emb_tensor
    }

# 保存为新的 pt 文件
torch.save(spk2info, 'data/train/spk2info.pt')
