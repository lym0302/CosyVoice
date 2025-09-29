# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Callable, Generator
from transformers import Qwen2ForCausalLM
import math

from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.snn.modules import mem_update_MSF
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.common import th_accuracy, IGNORE_ID


class SNNAdapter(nn.Module):
    """SNN适配器 - 在不修改原始Transformer结构的情况下增加SNN功能"""
    
    def __init__(self, hidden_size, msf_config=None):
        super().__init__()
        self.hidden_size = hidden_size
        
        # MSF SNN组件
        msf_config = msf_config or {}
        self.snn_processor = mem_update_MSF(
            decay=msf_config.get('decay', 0.25),
            init_thre=msf_config.get('init_thre', 1.0),
            D=msf_config.get('D', 4),
            surro_gate=msf_config.get('surro_gate', 'rectangular')
        )
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始时SNN影响很小
        
        # 可选的投影层，保持维度一致
        self.use_projection = msf_config.get('use_projection', False)
        if self.use_projection:
            self.proj_in = nn.Linear(hidden_size, hidden_size)
            self.proj_out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size]
        返回: SNN增强后的特征
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 可选的输入投影
        if self.use_projection:
            x_proj = self.proj_in(x)
        else:
            x_proj = x
        
        # MSF处理：需要 [time_window, batch, features] 格式
        snn_input = x_proj.permute(1, 0, 2)  # [seq_len, batch, hidden_size]
        snn_output = self.snn_processor(snn_input)  # MSF处理
        snn_output = snn_output.permute(1, 0, 2)  # 转回 [batch, seq_len, hidden_size]
        
        # 可选的输出投影
        if self.use_projection:
            snn_output = self.proj_out(snn_output)
        
        # 自适应融合：原始特征 + alpha * SNN特征
        enhanced_output = x + self.alpha * snn_output
        
        return enhanced_output


class Qwen2Encoder_SNN(nn.Module):
    """
    保持预训练权重兼容的SNN增强Qwen2编码器
    策略：在原始Transformer层之后插入SNN适配器
    """
    
    def __init__(self, pretrain_path, snn_layer_indices=None, msf_config=None):
        super().__init__()
        
        # 1. 加载原始预训练模型（结构不变）
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)
        self.config = self.model.config
        
        # 2. MSF配置
        self.msf_config = msf_config or {
            'decay': 0.25,
            'init_thre': 1.0,
            'D': 4,
            'surro_gate': 'rectangular',
            'use_projection': False
        }
        
        # 3. 确定哪些层添加SNN适配器
        if snn_layer_indices is None:
            # 默认：中间1/3的层使用SNN
            num_layers = len(self.model.model.layers)
            start_idx = num_layers // 3
            end_idx = start_idx + num_layers // 3
            self.snn_layer_indices = list(range(start_idx, end_idx))
        else:
            self.snn_layer_indices = snn_layer_indices
            
        # 4. 为指定层创建SNN适配器
        self.snn_adapters = nn.ModuleDict()
        for layer_idx in self.snn_layer_indices:
            self.snn_adapters[str(layer_idx)] = SNNAdapter(
                self.config.hidden_size, 
                self.msf_config
            )
            
        print(f"🧠 SNN适配器已添加到层: {self.snn_layer_indices}")
        print(f"📦 预训练权重保持完整，SNN适配器从零开始训练")
    
    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        B, T = xs.size(0), xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)  # [B, T] bool
        
        # 创建causal mask并扩展到batch size
        causal_mask = torch.tril(torch.ones((T, T), device=xs.device)).bool()
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, T, T)  # [B, 1, T, T]
        
        # 逐层前向传播
        hidden_states = xs
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # 标准Transformer层前向
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            
            # 如果该层有SNN适配器，则进行SNN增强
            if layer_idx in self.snn_layer_indices:
                snn_adapter = self.snn_adapters[str(layer_idx)]
                hidden_states = snn_adapter(hidden_states)
                
        return hidden_states, masks.unsqueeze(1)
    
    def forward_one_step(self, xs: torch.Tensor, masks=None, cache=None):
        """单步推理 - 需要处理SNN适配器的状态"""
        hidden_states = xs
        new_cache = []
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            past_kv = cache[layer_idx] if cache is not None else None
            
            # 处理attention mask
            attention_mask = None
            if masks is not None:
                # masks shape is [batch, seq_len, seq_len], need to expand for attention
                attention_mask = masks
            
            # 标准层前向
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=True
            )
            hidden_states = layer_outputs[0]
            new_cache.append(layer_outputs[1] if len(layer_outputs) > 1 else None)
            
            # SNN增强（单步推理时需要小心处理）
            if layer_idx in self.snn_layer_indices:
                snn_adapter = self.snn_adapters[str(layer_idx)]
                hidden_states = snn_adapter(hidden_states)
        
        return hidden_states, new_cache
    
    def get_snn_strength(self):
        """获取当前SNN适配器的影响强度"""
        strengths = {}
        for layer_idx in self.snn_layer_indices:
            alpha = self.snn_adapters[str(layer_idx)].alpha.item()
            strengths[layer_idx] = alpha
        return strengths
    
    def set_snn_strength(self, alpha_value):
        """设置所有SNN适配器的影响强度"""
        for layer_idx in self.snn_layer_indices:
            self.snn_adapters[str(layer_idx)].alpha.data.fill_(alpha_value)
    
    def freeze_pretrained_weights(self):
        """冻结预训练权重，只训练SNN适配器"""
        # 冻结原始模型权重
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻SNN适配器权重    
        for adapter in self.snn_adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True
                
        print("🔒 预训练权重已冻结，只有SNN适配器可训练")
    
    def unfreeze_all_weights(self):
        """解冻所有权重，进行端到端微调"""
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 所有权重已解冻，可进行端到端训练")


class Qwen2LM_SNN(Qwen2LM):
    """兼容预训练权重的SNN增强Qwen2语言模型"""
    
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: Qwen2Encoder_SNN,  # 注意这里是新的编码器
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
        training_mode: str = 'snn_only',  # 'snn_only', 'end_to_end', 'gradual'
    ):
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
            mix_ratio=mix_ratio
        )
        
        self.training_mode = training_mode
        self._setup_training_mode()
        
        print(f"🎯 SNN-LLM兼容模型初始化完成")
        print(f"   - 训练模式: {training_mode}")
        print(f"   - SNN层位置: {llm.snn_layer_indices}")
        print(f"   - 预训练权重: 完全保留")
    
    def _setup_training_mode(self):
        """根据训练模式设置权重冻结状态"""
        if self.training_mode == 'snn_only':
            # 只训练SNN适配器
            self.llm.freeze_pretrained_weights()
        elif self.training_mode == 'end_to_end':
            # 端到端训练
            self.llm.unfreeze_all_weights()
        elif self.training_mode == 'gradual':
            # 渐进训练：先冻结，后面可以解冻
            self.llm.freeze_pretrained_weights()
    
    def set_training_mode(self, mode: str):
        """动态切换训练模式"""
        self.training_mode = mode
        self._setup_training_mode()
        print(f"🔄 训练模式已切换为: {mode}")
    
    def get_training_status(self):
        """获取当前训练状态"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        snn_strengths = self.llm.get_snn_strength()
        
        return {
            'training_mode': self.training_mode,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'snn_strengths': snn_strengths
        }
    
    def forward(self, batch: dict, device: torch.device):
        """前向传播 - 与原版完全兼容"""
        return super().forward(batch, device)
    
    @torch.inference_mode()
    def inference(self, *args, **kwargs):
        """推理 - 与原版完全兼容"""
        return super().inference(*args, **kwargs)
    
    def progressive_training_schedule(self, epoch: int, total_epochs: int):
        """渐进式训练调度"""
        if self.training_mode == 'gradual':
            # 前50%训练周期只训练SNN适配器
            if epoch < total_epochs * 0.5:
                self.llm.freeze_pretrained_weights()
                # 逐渐增加SNN强度
                alpha = min(0.5, epoch / (total_epochs * 0.3))
                self.llm.set_snn_strength(alpha)
            else:
                # 后50%解冻所有权重，进行端到端微调
                self.llm.unfreeze_all_weights()
                # SNN强度保持稳定
                self.llm.set_snn_strength(0.3)