# -*- coding: utf-8 -*-
"""
Configuration for training hyperparameters, data paths, and model parameters
"""

class Config:
    # Data paths
    data_dir = "data/train.txt"  # Training text path
    vocab_path = "data/char_to_idx.json"        # Dictionary file path
    
    # Model parameters
    d_model = 1024          # Model dimension
    num_layers = 8          # Number of Transformer layers
    num_heads = 8           # Number of attention heads
    d_ff = 4096             # Feed-forward network dimension
    max_seq_len = 512       # Maximum sequence length
    dropout = 0.1           # Dropout probability
    
    # Training parameters
    batch_size = 64       # Batch size
    learning_rate = 1e-4    # Learning rate
    num_epochs = 10         # Number of training epochs
    device = "cuda"         # Training device ("cuda" or "cpu")
    
    # Checkpoint continuation parameters
    continue_training = False               # Whether to continue training from checkpoint
    load_checkpoint_path = "checkpoints/char_transformer_epoch0.pt"  # Path to load model from
    
config = Config()