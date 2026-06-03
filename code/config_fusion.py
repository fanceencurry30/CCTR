# Configuration parameter file

# ============================================
# 📌 Model Architecture Parameters
# ============================================
FEATURE_DIM = 100  # Dimension of top 100 candidate characters
NUM_HEADS = 8      # Number of attention heads
HIDDEN_DIM = 512   # FFN hidden dimension (Increased from 256→512 to enhance expressive capability)
DROPOUT = 0.1      # Dropout ratio (Standard value, suitable for Transformer)
NUM_ENCODER_LAYERS = 6  # Transformer encoder layers
NUM_DECODER_LAYERS = 6  # Transformer decoder layers

# ============================================
# 🎯 Training Hyperparameters (Based on modern best practices)
# ============================================
BATCH_SIZE = 128   # Increase batch size to improve training stability
EPOCHS = 200       # Sufficient training epochs

# Learning rate strategy: Cosine Annealing with Warmup (Same as BERT/GPT)
LEARNING_RATE = 1e-4      # Initial learning rate (peak after warmup)
WARMUP_EPOCHS = 10        # Warmup epochs
MIN_LEARNING_RATE = 1e-6  # Minimum learning rate (end of cosine decay)

# Optimizer configuration
WEIGHT_DECAY = 0.01       # AdamW weight decay
BETA1 = 0.9               # Adam beta1
BETA2 = 0.98              # Adam beta2 (Recommended value from Transformer paper)
EPS = 1e-9                # Adam epsilon (numerical stability)

# Label Smoothing (prevents overfitting)
LABEL_SMOOTHING = 0.1     # Label smoothing coefficient

# Gradient Clipping (prevents gradient explosion)
GRAD_CLIP = 1.0           # Gradient clipping threshold

# ============================================
# 📁 File Paths
# ============================================
FUSION_MODEL_SAVE_PATH = "models_nrtr_image_3/fusion_model"