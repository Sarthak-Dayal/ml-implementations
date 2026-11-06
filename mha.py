# Nov 5, 2026. Simple MHA implementation in PyTorch
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    
    assert d_model % num_heads == 0
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_head = d_model // num_heads

    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

  def forward(self, x):
    B, T, D = x.shape
    
    q = self.q_proj(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
    k = self.k_proj(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
    v = self.v_proj(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
    
    attn_scores = (q @ k.mT) / (self.d_head**0.5)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    combined = (attn_weights @ v).transpose(1,2).reshape(B, T, D)
    
    return self.out_proj(combined)