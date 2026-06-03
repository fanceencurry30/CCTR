# model.py
# Transformer Character Filling Language Model
# Contains StandardAttention (default) and LinformerAttention (optional)

import torch
import torch.nn as nn
from config import *


class StandardAttention(nn.Module):
    """Standard Multi-Head Self-Attention Mechanism"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.output_proj(out)
        
        return out


class LinformerAttention(nn.Module):
    """Linformer Attention Mechanism (reduces computation by 50%)"""
    def __init__(self, embed_dim, num_heads, k_dim=64, dropout=0.1, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_dim = k_dim
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.k_compress = nn.Linear(max_seq_len, k_dim, bias=False)
        self.v_compress = nn.Linear(max_seq_len, k_dim, bias=False)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if seq_len < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - seq_len
            k = torch.cat([k, torch.zeros(batch_size, pad_len, self.embed_dim, device=x.device)], dim=1)
            v = torch.cat([v, torch.zeros(batch_size, pad_len, self.embed_dim, device=x.device)], dim=1)
        
        k = self.k_compress(k.transpose(1, 2)).transpose(1, 2)
        v = self.v_compress(v.transpose(1, 2)).transpose(1, 2)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.output_proj(out)
        
        return out


class TransformerLM(nn.Module):
    """Transformer Language Model"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, dropout=DROPOUT, use_linformer=False, k_dim=SPARSE_QUERIES):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, embed_dim))
        
        # Select attention mechanism
        if use_linformer:
            # print(f"Using Linformer attention (k_dim={k_dim})")
            self.attention = nn.ModuleList([
                LinformerAttention(embed_dim, num_heads, k_dim, dropout) for _ in range(num_layers)
            ])
        else:
            # print(f"Using standard multi-head attention")
            self.attention = nn.ModuleList([
                StandardAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)
            ])
        
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.attn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.output = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)
    
    def forward(self, x, mask=None):
        embeds = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        embeds = self.dropout(embeds)
        
        for attn, ff, attn_norm, ffn_norm in zip(
            self.attention, self.feed_forward, self.attn_norm, self.ffn_norm
        ):
            attn_out = attn(attn_norm(embeds), mask)
            embeds = embeds + attn_out
            
            ff_out = ff(ffn_norm(embeds))
            embeds = embeds + ff_out
        
        logits = self.output(embeds)
        return logits