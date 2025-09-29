# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.utils.mask import make_pad_mask

class MultiModalPitchPredictor(nn.Module):
    """
    Multi-modal pitch predictor that takes text embeddings, speech token embeddings, 
    and speaker embeddings as input to predict pitch sequence.
    """
    def __init__(self,
                 llm: Qwen2LM,
                 text_dim: int = 1024,
                 speech_token_dim: int = 1024, 
                 spk_dim: int = 192,
                 hidden_dim: int = 512,
                 attention_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.llm = llm
        for param in self.llm.parameters():
            param.requires_grad = False
        self.text_dim = text_dim
        self.speech_token_dim = speech_token_dim
        assert self.text_dim == self.speech_token_dim, "text_dim should be same as speech_token_dim!!!"
        self.spk_dim = spk_dim
        self.hidden_dim = hidden_dim
        
        # Speaker embedding projection to match speech token dimension
        self.spk_projection = nn.Linear(spk_dim, speech_token_dim)
        
        # Text to speech token cross attention
        # This will project text features to align with speech token sequence length
        self.text_speech_cross_attention = MultiheadAttention(
            embed_dim=speech_token_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion layer
        # Input: speech_token_emb + projected_spk_emb + cross_attended_text_token_emb
        self.fusion_layer = nn.Sequential(
            nn.Linear(speech_token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Convolutional layers for temporal modeling (similar to original f0_predictor)
        self.conv_layers = nn.Sequential(
            # 1️⃣ 上采样：反卷积，长度翻倍
            weight_norm(nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)),
            nn.ELU(),

            # 2️⃣ 平滑卷积：抑制棋盘格伪影
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.ELU(),

            # 3️⃣ 后续处理
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        
        # Final pitch prediction head
        self.pitch_head = nn.Linear(hidden_dim, 1)
        

    def forward(self, batch:dict, device: torch.device,):
        """
        Returns:
            pitch: Predicted pitch sequence [B, speech_token_len]
        """
        # get emb
        text_token_emb, speech_token_emb = self.llm.get_emb(batch, device)
        batch_size, speech_len, _ = speech_token_emb.shape
        spk_emb = batch['spk_embedding'].to(device)
        text_token_emb = text_token_emb.to(device)
        speech_token_emb = speech_token_emb.to(device)
    
        # Cross attention: align text features with speech token sequence
        # Query: speech_token_emb, Key&Value: text_projected
        text_token_len = batch['text_token_len'].to(device)
        text_mask = make_pad_mask(text_token_len).to(device)  # bool, True 表示 padding
        # print("1111111111111111: ", text_mask.shape)
        text_attended, _ = self.text_speech_cross_attention(
            query=speech_token_emb,      # [B, speech_len, speech_token_dim]
            key=text_token_emb,          # [B, text_len, speech_token_dim] 
            value=text_token_emb,        # [B, text_len, speech_token_dim]
            key_padding_mask=text_mask  # mask text padding
        )  # Output: [B, speech_len, speech_token_dim]
        # print("222222222222222222: ", text_attended.shape)
        # Fusion: combine all three modalities
        spk_projected = self.spk_projection(spk_emb)  # [B, speech_token_dim]
        spk_expanded = spk_projected.unsqueeze(1).expand(batch_size, speech_len, -1)  # [B, speech_len, speech_token_dim]
        fused_emb = speech_token_emb + spk_expanded + text_attended  # [B, speech_len, speech_token_dim]
        fused_features = self.fusion_layer(fused_emb)  # [B, speech_len, hidden_dim]
        # print("3333333333333333333333333: ", fused_features.shape)
        # Transpose for conv1d: [B, speech_len, hidden_dim] -> [B, hidden_dim, speech_len]
        conv_input = fused_features.transpose(1, 2)
        conv_output = self.conv_layers(conv_input)  # [B, hidden_dim, speech_len]
        conv_output = conv_output.transpose(1, 2)
        # print("44444444444444444444444: ", conv_output.shape)
        #  Final pitch prediction
        predicted_pitch = torch.abs(self.pitch_head(conv_output).squeeze(-1))  # [B, speech_len]
        # print("55555555555555555555: ", pitch.shape)
        
        target_pitch = batch.get('pitch_feat').to(device)  # [B, speech_len]
        pitch_feat_len = batch.get('pitch_feat_len').to(device)
        pitch_mask = ~make_pad_mask(pitch_feat_len).to(device)  # True = 有效位置
        loss = torch.nn.functional.l1_loss(predicted_pitch[pitch_mask], target_pitch[pitch_mask])

        return {"loss": loss}
    
    @torch.inference_mode()
    def inference(self, text_token, speech_token, spk_emb, device: torch.device,):
        """
        Returns:
            pitch: Predicted pitch sequence [B, speech_token_len]
        """
        batch = {}
        batch['text_token'] = text_token
        batch['speech_token'] = speech_token
        # get emb
        text_token_emb, speech_token_emb = self.llm.get_emb(batch, device)
        batch_size, speech_len, _ = speech_token_emb.shape
        spk_emb = spk_emb.to(device)
        text_token_emb = text_token_emb.to(device)
        speech_token_emb = speech_token_emb.to(device)

        # Cross attention: align text features with speech token sequence
        # Query: speech_token_emb, Key&Value: text_projected
        text_token_len = torch.tensor([text_token.shape[1]]).to(device)
        text_mask = make_pad_mask(text_token_len).to(device)  # bool, True 表示 padding
        text_attended, _ = self.text_speech_cross_attention(
            query=speech_token_emb,      # [B, speech_len, speech_token_dim]
            key=text_token_emb,          # [B, text_len, speech_token_dim] 
            value=text_token_emb,        # [B, text_len, speech_token_dim]
            key_padding_mask=text_mask  # mask text padding
        )  # Output: [B, speech_len, speech_token_dim]
        
        # Fusion: combine all three modalities
        spk_projected = self.spk_projection(spk_emb)  # [B, speech_token_dim]
        spk_expanded = spk_projected.unsqueeze(1).expand(batch_size, speech_len, -1)  # [B, speech_len, speech_token_dim]
        fused_emb = speech_token_emb + spk_expanded + text_attended  # [B, speech_len, speech_token_dim]
        fused_features = self.fusion_layer(fused_emb)  # [B, speech_len, hidden_dim]
        
        # Transpose for conv1d: [B, speech_len, hidden_dim] -> [B, hidden_dim, speech_len]
        conv_input = fused_features.transpose(1, 2)
        conv_output = self.conv_layers(conv_input)  # [B, hidden_dim, speech_len]
        conv_output = conv_output.transpose(1, 2)
        
        #  Final pitch prediction
        predicted_pitch = torch.abs(self.pitch_head(conv_output).squeeze(-1))  # [B, speech_len]
        
        return predicted_pitch
