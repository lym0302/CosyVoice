# lora_core.py
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    在现有 Linear 上叠加 LoRA: y = xW^T + (x A^T B^T) * (alpha/r) + bias
    - 冻结原始 W, bias（可选）
    - 仅训练 A, B
    - 可选择合并为权重做推理
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        init_scale: float = 1e-4,
        train_bias: bool = False,   # 一般不训练 bias
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias is not None

        # 冻结原始权重与bias
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        self.bias_param = None
        if self.bias:
            self.bias_param = nn.Parameter(base_linear.bias.data.clone(), requires_grad=train_bias)

        # LoRA 部分
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        if r > 0:
            # A: (r, in_features), B: (out_features, r)
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            # 推荐小范围初始化，避免一开始就破坏原分布
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # 可选更小初值
            self.lora_A.data.mul_(init_scale)
            self.lora_B.data.mul_(init_scale)
        else:
            # 允许 r=0 当作占位，等同于不启用LoRA
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.merged = False  # 是否已将 LoRA 合并进权重

    def forward(self, x):
        # 基础线性
        out = torch.nn.functional.linear(x, self.weight, self.bias_param)

        # 叠加 LoRA 分支
        if self.r > 0 and not self.merged:
            x_d = self.lora_dropout(x)                # [B, T, in]
            # x @ A^T -> [B, T, r] ; 再 @ B^T -> [B, T, out]
            lora_out = x_d @ self.lora_A.t()
            lora_out = lora_out @ self.lora_B.t()
            out = out + lora_out * self.scaling

        return out

    @torch.no_grad()
    def merge_lora(self):
        """把 LoRA 合并进权重（推理时用），之后计算等价于普通 Linear"""
        if self.r == 0 or self.merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]
        self.weight.data += delta_w
        self.merged = True

    @torch.no_grad()
    def unmerge_lora(self):
        """反合并，恢复原权重（通常不需要）"""
        if self.r == 0 or not self.merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data -= delta_w
        self.merged = False


# lora_utils.py
import torch
import torch.nn as nn
from typing import Iterable, List, Tuple, Dict

def apply_lora_to_named_linears(
    model: nn.Module,
    target_names: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    train_bias: bool = False,
) -> nn.Module:
    """
    把 model 中名字匹配的 nn.Linear 替换为 LoRALinear。
    target_names: 例如 ["llm_decoder"] 或 ["llm_decoder", "some_layer.out_proj"]
    """
    name_to_module = dict(model.named_modules())
    for name in list(name_to_module.keys()):
        if name in target_names:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.split(".")[-1]
            parent = name_to_module[parent_name] if parent_name else model
            linear = getattr(parent, child_name)
            assert isinstance(linear, nn.Linear), f"{name} is not nn.Linear"
            setattr(parent, child_name,
                    LoRALinear(linear, r=r, alpha=alpha, dropout=dropout, train_bias=train_bias))
    return model

def only_optimize_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """只返回 LoRA 可训练参数（A/B 以及可选 bias）"""
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n) or ("bias_param" in n and p.requires_grad):
            yield p

def freeze_non_lora(model: nn.Module):
    """冻结除 LoRA 以外的全部参数"""
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n) or ("bias_param" in n and p.requires_grad):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """仅导出 LoRA 参数（便于轻量保存）"""
    state = {}
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n) or ("bias_param" in n and p.requires_grad):
            state[n] = p.detach().cpu()
    return state

def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = False):
    """加载 LoRA 参数（在已 apply_lora 之后）"""
    own = dict(model.named_parameters())
    missing, unexpected = [], []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].data.copy_(v.to(own[k].device))
        else:
            if strict:
                unexpected.append(k)
    if strict:
        # 找缺失
        for k in own.keys():
            if (("lora_A" in k) or ("lora_B" in k)) and k not in state:
                missing.append(k)
        if missing or unexpected:
            raise RuntimeError(f"Missing: {missing}, Unexpected: {unexpected}")

def merge_all_lora(model: nn.Module):
    """递归合并所有 LoRALinear 的 LoRA 权重"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_lora()
