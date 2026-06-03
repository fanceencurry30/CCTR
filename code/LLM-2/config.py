# Model parameters
# Note: The actual vocabulary size should be dynamically obtained from CharVocab
# Original 9633 + 3 new special tokens ([PAD], [SOS], [EOS]) = 9636 (□ is already in the original vocabulary)
VOCAB_SIZE = 9636  # Vocabulary size (including special tokens)
EMBED_DIM = 512     # Embedding dimension
HIDDEN_DIM = 3072   # Feedforward network hidden dimension
NUM_LAYERS = 24     # Number of Transformer layers
NUM_HEADS = 8       # Number of attention heads
SPARSE_QUERIES = 64 # Number of sparse attention query vectors
DROPOUT = 0.1       # Dropout rate

# Training parameters
BATCH_SIZE = 32      # Batch size per GPU (reduces memory usage)
ACCUMULATION_STEPS = 4  # Gradient accumulation steps (effective batch = 32×4×4GPUs = 512)
EPOCHS = 50         # Number of training epochs
LEARNING_RATE = 1e-4 # Learning rate
MAX_SEQ_LEN = 128   # Maximum sequence length
NOISE_P = 0.02      # Noise ratio for non-masked characters

# Memory optimization
USE_MIXED_PRECISION = True  # Mixed precision training (saves 50% memory, speeds up by 30%)

# Data paths
DICTIONARY_PATH = "data/char_to_idx.json"  # Dictionary file path
TRAIN_DATA_PATH = "data/train.txt"         # Training data path
MODEL_SAVE_PATH = "models/models.pth"  # Model save path
CACHE_PATH = "data/train_cache.pkl"        # Preprocessed data cache path

# Special tokens
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "□"    # Mask token (blank character)
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"