# Improved SNN-Enhanced CosyVoice
# 针对训练稳定性和效果的改进版本

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.llm.llm import Qwen2LM


class StabilizedSNNAdapter(nn.Module):
    """
    稳定化的SNN适配器 - 解决训练不稳定问题
    """
    
    def __init__(self, hidden_size, msf_config):
        super().__init__()
        from cosyvoice.snn.modules import mem_update_MSF
        
        # SNN处理器
        self.snn_processor = mem_update_MSF(
            decay=msf_config.get('decay', 0.25),
            init_thre=msf_config.get('init_thre', 1.0),
            D=msf_config.get('D', 4),
            surro_gate=msf_config.get('surro_gate', 'rectangular')
        )
        
        # 关键改进1: 梯度缩放器（增加初始值确保有效梯度传播）
        self.grad_scale = nn.Parameter(torch.tensor(0.5))
        
        # 关键改进2: 自适应融合权重（从适中值开始，确保梯度流）
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 增加初始值，确保有效的梯度流
        
        # 关键改进3: 预处理层（减少输入幅度）  
        self.input_norm = nn.LayerNorm(hidden_size)
        self.pre_proj = nn.Linear(hidden_size, hidden_size)
        
        # 关键改进4: 后处理层（稳定输出）
        self.post_norm = nn.LayerNorm(hidden_size) 
        self.post_proj = nn.Linear(hidden_size, hidden_size)
        
        # 关键改进5: Dropout防止过拟合
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """更积极的权重初始化，确保梯度流"""
        nn.init.xavier_uniform_(self.pre_proj.weight, gain=0.5)  # 增加gain值
        nn.init.xavier_uniform_(self.post_proj.weight, gain=0.5)  # 增加gain值
        nn.init.zeros_(self.pre_proj.bias)
        nn.init.zeros_(self.post_proj.bias)
    
    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 1. 输入预处理和归一化
        x_norm = self.input_norm(x)
        x_proj = self.pre_proj(x_norm)
        
        # 2. SNN处理（输入格式转换）
        snn_input = x_proj.permute(1, 0, 2)  # [seq_len, batch, hidden_size]
        
        # 关键改进: 梯度裁剪保护（放松裁剪阈值）
        with torch.no_grad():
            input_norm = torch.norm(snn_input)
            if input_norm > 50.0:  # 提高阈值，减少过度裁剪
                snn_input = snn_input * (50.0 / input_norm)
        
        snn_output = self.snn_processor(snn_input)  
        snn_output = snn_output.permute(1, 0, 2)  # 转回 [batch, seq_len, hidden_size]
        
        # 3. 后处理和归一化
        snn_output = self.post_norm(snn_output)
        snn_output = self.post_proj(snn_output)
        snn_output = self.dropout(snn_output)
        
        # 4. 改进的残差连接（确保有效的梯度流）
        alpha_clamped = torch.clamp(self.alpha, 0.0, 0.5)  # 提高上限，允许更强的融合
        grad_scale_clamped = torch.clamp(self.grad_scale, 0.1, 1.0)  # 确保梯度缩放有效
        enhanced_output = x + alpha_clamped * snn_output * grad_scale_clamped
        
        return enhanced_output
    
    def get_fusion_weight(self):
        """返回当前融合权重，用于监控"""
        return torch.clamp(self.alpha, 0.0, 0.5).item()


class Qwen2Encoder_SNN_Improved(nn.Module):
    """
    改进版SNN增强Qwen2编码器 - 注重训练稳定性
    """
    
    def __init__(self, pretrain_path, snn_layer_indices=None, msf_config=None):
        super().__init__()
        
        # 加载预训练模型
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)
        
        # SNN配置
        self.snn_layer_indices = snn_layer_indices or [10, 11, 12, 13]
        msf_config = msf_config or {}
        
        # 创建稳定化SNN适配器
        self.snn_adapters = nn.ModuleDict()
        for layer_idx in self.snn_layer_indices:
            if layer_idx < len(self.model.model.layers):
                adapter = StabilizedSNNAdapter(
                    hidden_size=self.model.config.hidden_size,
                    msf_config=msf_config
                )
                self.snn_adapters[str(layer_idx)] = adapter
        
        print(f"🧠 改进版: 稳定化SNN适配器已添加到层: {self.snn_layer_indices}")
        print(f"📈 使用梯度缩放、权重裁剪和保守融合策略")
        
    def freeze_pretrained_weights(self):
        """冻结预训练权重，只训练SNN组件"""
        frozen_params = 0
        snn_params = 0
        
        for name, param in self.named_parameters():
            if any(snn_name in name for snn_name in ['snn_adapters', 'alpha', 'grad_scale', 'pre_proj', 'post_proj']):
                param.requires_grad = True
                snn_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"🔒 改进版: 预训练权重已冻结: {frozen_params:,} 参数")
        print(f"🎯 SNN可训练参数: {snn_params:,} 参数")
        return snn_params, frozen_params
    
    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        B, T = xs.size(0), xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)  # [B, T] bool
        
        # 创建causal mask并扩展到batch size
        causal_mask = torch.tril(torch.ones((T, T), device=xs.device)).bool()
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, T, T)
        
        # 逐层前向传播
        hidden_states = xs
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # 标准Transformer层前向
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            
            # 如果该层有SNN适配器，则进行SNN增强
            if layer_idx in self.snn_layer_indices:
                snn_adapter = self.snn_adapters[str(layer_idx)]
                
                # 关键改进: 输入数值稳定性检查
                if self.training:
                    with torch.no_grad():
                        # 检查输入的数值稳定性而不是梯度
                        input_norm = torch.norm(hidden_states)
                        if torch.isnan(input_norm) or torch.isinf(input_norm) or input_norm > 100.0:
                            # 如果输入不稳定，跳过SNN处理
                            continue
                
                hidden_states = snn_adapter(hidden_states)
                
        return hidden_states, masks.unsqueeze(1)
    
    def forward_one_step(self, xs, masks, cache=None):
        """Single-step forward for inference with caching"""
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache
    
    def get_training_status(self):
        """返回训练状态信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 收集SNN适配器状态
        snn_weights = {}
        for idx, adapter in self.snn_adapters.items():
            snn_weights[idx] = adapter.get_fusion_weight()
        
        return {
            'training_mode': 'snn_improved',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'snn_weights': snn_weights
        }


class Qwen2LM_SNN_Improved(Qwen2LM):
    """
    改进版SNN增强的Qwen2语言模型 - 稳定训练版本
    """
    
    def __init__(self, llm_input_size, llm_output_size, speech_token_size, 
                 length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15],
                 training_mode='snn_only', llm=None, sampling=None):
        
        # 使用改进的SNN编码器
        if llm is None:
            llm = Qwen2Encoder_SNN_Improved(
                pretrain_path='',  # 会在配置中指定
                snn_layer_indices=[10, 11, 12, 13],
                msf_config={'decay': 0.25, 'init_thre': 1.0, 'D': 4}
            )
        
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size, 
            speech_token_size=speech_token_size,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
            mix_ratio=mix_ratio,
            llm=llm,
            sampling=sampling
        )
        
        self.training_mode = training_mode
        
        # 根据训练模式设置参数梯度
        if training_mode == 'snn_only':
            self.llm.freeze_pretrained_weights()
            print(f"🎯 改进版: SNN-only稳定训练模式")
        else:
            print(f"🔄 改进版: {training_mode}训练模式")
    
    def get_training_status(self):
        """获取训练状态"""
        return self.llm.get_training_status()