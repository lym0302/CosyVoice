import torch
import sys

# 加载原始 checkpoint
# raw_model_path = "exp/cosyvoice/llm/torch_ddp/epoch_2_whole.pt"
# new_model_path = "exp/cosyvoice/llm/torch_ddp/epoch_2_whole_infer.pt"
# epoch = sys.argv[1]
# exp_dir = sys.argv[2]
# model = sys.argv[3]
# raw_model_path = f"{exp_dir}/cosyvoice/{model}/torch_ddp/epoch_{epoch}_whole.pt"
# new_model_path = f"{exp_dir}/cosyvoice/{model}/torch_ddp/epoch_{epoch}_whole_infer.pt"

raw_model_path = sys.argv[1]
# new_model_path = sys.argv[2]
new_model_path = raw_model_path.replace(".pt", "_infer.pt")

ckpt = torch.load(raw_model_path, map_location="cpu")

# 判断是否包含模型的 key
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt  # 有可能直接就是 state_dict

# 过滤掉非模型参数
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('epoch') and not k.startswith('step')}

# 保存新的权重文件
torch.save(filtered_state_dict, new_model_path)

print(f"Saved cleaned model to {new_model_path}")
