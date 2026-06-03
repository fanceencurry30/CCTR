#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion Model - Clean Version (Transformer-based Architecture)

Core Architecture:
1. Sequence Embedding
2. Three Independent Transformer Encoders (Query/Key/Value)
3. Bidirectional Cross Attention (OCR ↔ LM)
4. Bidirectional GRU (Sequence Modeling)
5. Multi-layer Decoder (Self-Attn + Conv1D + GLU + FFN)
6. Temperature-Scaled Output Layer

Design Principles:
- Input data is pre-normalized (sum of probabilities = 1.0)
- No truncation or additional preprocessing
- Model outputs logits, with softmax handled by the loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross Attention Fusion Model (Transformer-based OCR+LM Fusion)
    """
    def __init__(self, feature_dim=100, num_heads=8, hidden_dim=768, dropout=0.1, 
                 num_encoder_layers=6, num_decoder_layers=6, temperature=0.5):
        """
        Args:
            feature_dim: Input feature dimension (number of top-k candidates, default: 100)
            num_heads: Number of attention heads (default: 8, embed_dim=512 must be divisible by this)
            hidden_dim: FFN hidden layer dimension (default: 768)
            dropout: Dropout probability (default: 0.1)
            num_encoder_layers: Number of Encoder layers (default: 6)
            num_decoder_layers: Number of Decoder layers (default: 6)
            temperature: Softmax temperature parameter (default: 0.5)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embed_dim = 512  # Fixed embedding dimension, consistent with fusion_model_60p.py
        self.temperature = temperature
        
        print(f"📊 CrossAttentionFusion Configuration:")
        print(f"  - Input Dimension: {feature_dim}")
        print(f"  - Embedding Dimension: {self.embed_dim}")
        print(f"  - Number of Attention Heads: {num_heads}")
        print(f"  - FFN Hidden Layer: {hidden_dim}")
        print(f"  - Number of Encoder Layers: {num_encoder_layers}")
        print(f"  - Number of Decoder Layers: {num_decoder_layers}")
        print(f"  - Temperature Parameter: {temperature}")
        print(f"  - Dropout: {dropout}")
        print(f"  ✅ Input data is pre-normalized (no truncation)")
        
        # ============ Sequence Embedding Layer ============
        # Embed 1D probability values into embed_dim-dimensional space
        # Input: [batch, 100, 1] → Output: [batch, 100, embed_dim]
        self.seq_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(dropout)
        )
        
        # ============ Three Independent Transformer Encoders ============
        # Encode query, key, value features separately
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.query_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.key_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.value_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # ============ Bidirectional Cross Attention ============
        # 1. LM queries OCR (LM attends to OCR)
        # 2. OCR queries LM (OCR attends to LM)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.bi_cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.embed_dim)
        
        # ============ Bidirectional GRU ============
        # 4-layer bidirectional GRU to capture sequence dependencies
        self.gru = nn.GRU(
            self.embed_dim,
            self.embed_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # ============ Decoder Module ============
        # Each layer includes: self-attention + conv1d + GLU + FFN
        decoder_layer = nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True
            ),
            'conv1d': nn.Conv1d(self.embed_dim, self.embed_dim * 2, kernel_size=3, padding=1),
            'conv_proj': nn.Conv1d(self.embed_dim * 2, self.embed_dim, kernel_size=1),
            'conv_norm': nn.LayerNorm(self.embed_dim),
            'glu': nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.GLU(),
            ),
            'ffn': nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.Dropout(dropout)
            ),
            'norm1': nn.LayerNorm(self.embed_dim),
            'norm2': nn.LayerNorm(self.embed_dim),
            'norm3': nn.LayerNorm(self.embed_dim),
            'norm4': nn.LayerNorm(self.embed_dim)
        })
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])
        
        # ============ Output Layer ============
        # Map embed_dim-dimensional features back to feature_dim-dimensional probability distribution
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, feature_dim)
        )
    
    def forward(self, ocr_probs, lm_probs):
        """
        Forward Propagation
        
        Args:
            ocr_probs: [batch, 1, 100] OCR candidate probabilities (normalized, sum=1.0)
            lm_probs:  [batch, 1, 100] LM candidate probabilities (normalized, sum=1.0)
        
        Returns:
            scaled_logits: [batch, 100] Fusion logits (without softmax, for loss function)
        """
        batch_size = ocr_probs.size(0)
        
        # ============ Phase 1: Reshape to Sequence ============
        # Reshape probability distributions into sequence format
        # Data is pre-normalized during preprocessing, used directly
        ocr_seq = ocr_probs.view(batch_size, self.feature_dim, 1)  # [batch, 100, 1]
        lm_seq = lm_probs.view(batch_size, self.feature_dim, 1)  # [batch, 100, 1]
        
        # ============ Phase 2: Sequence Embedding ============
        # Embed 1D probability values into high-dimensional space
        ocr_seq = self.seq_embed(ocr_seq)  # [batch, 100, embed_dim]
        lm_seq = self.seq_embed(lm_seq)    # [batch, 100, embed_dim]
        
        # ============ Phase 3: Three Independent Encoders ============
        # LM serves as query, OCR serves as key/value
        query_feat = self.query_encoder(lm_seq)    # [batch, 100, embed_dim]
        key_feat = self.key_encoder(ocr_seq)       # [batch, 100, embed_dim]
        value_feat = self.value_encoder(ocr_seq)   # [batch, 100, embed_dim]
        
        # ============ Phase 4: Bidirectional Cross Attention ============
        # Direction 1: query(LM) attends to key/value(OCR)
        attn_output1, _ = self.cross_attention(query_feat, key_feat, value_feat)
        # Direction 2: query(OCR) attends to key/value(LM) - note: value still uses OCR
        attn_output2, _ = self.bi_cross_attention(key_feat, query_feat, value_feat)
        # Bidirectional fusion (average)
        attn_output = self.cross_norm((attn_output1 + attn_output2) / 2)
        
        # ============ Phase 5: Bidirectional GRU ============
        # Capture sequence dependencies
        gru_output, _ = self.gru(attn_output)  # [batch, 100, embed_dim*2]
        # Merge bidirectional outputs: forward + backward
        gru_output = gru_output[:, :, :self.embed_dim] + gru_output[:, :, self.embed_dim:]
        
        # ============ Phase 6: Decoder ============
        # Multi-layer decoder processing
        fusion = gru_output
        for layer in self.decoder:
            # 6.1 Self Attention
            self_attn_out, _ = layer['self_attn'](fusion, fusion, fusion)
            fusion = layer['norm1'](fusion + self_attn_out)
            
            # 6.2 Conv1D
            conv_out = layer['conv1d'](fusion.transpose(1, 2))  # [batch, embed_dim*2, 100]
            conv_out = F.gelu(conv_out)
            conv_out = layer['conv_proj'](conv_out).transpose(1, 2)  # [batch, 100, embed_dim]
            conv_out = layer['conv_norm'](conv_out)
            fusion = layer['norm2'](fusion + conv_out)
            
            # 6.3 GLU
            glu_out = layer['glu'](fusion)
            fusion = layer['norm3'](fusion + glu_out)
            
            # 6.4 FFN
            ffn_out = layer['ffn'](fusion)
            fusion = layer['norm4'](fusion + ffn_out)
        
        # ============ Phase 7: Output Layer ============
        # Global average pooling
        fusion = fusion.mean(dim=1)  # [batch, embed_dim]
        
        # Output mapping
        logits = self.output(fusion)  # [batch, 100]
        
        # Apply temperature scaling (on logits)
        # Temperature scaling makes the model output smoother or sharper
        # Note: Return logits now, softmax is handled by the loss function
        scaled_logits = logits / self.temperature
        
        return scaled_logits