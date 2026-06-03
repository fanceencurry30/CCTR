# -*- coding: utf-8 -*-
"""
Define CharTransformer model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=1024, num_layers=8, num_heads=8, d_ff=4096, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.char_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        # initial param
        nn.init.xavier_normal_(self.char_embed.weight)
        nn.init.xavier_normal_(self.pos_embed.weight)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        x_emb = self.char_embed(x) + self.pos_embed(positions)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x_emb = layer(x_emb, src_mask=mask)
        logits = self.output(x_emb)
        return logits