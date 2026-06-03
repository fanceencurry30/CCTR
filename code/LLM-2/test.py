import torch
import numpy as np
import random
import argparse
from model import TransformerLM
from data_loader import CharVocab
from utils import compute_n95, get_top_n_candidates
from config import *


def test_model(text, model_path=MODEL_SAVE_PATH, use_linformer=False):
    print("=" * 70)
    print("Model Testing")
    print("=" * 70)
    
    # Load vocabulary
    vocab = CharVocab(DICTIONARY_PATH)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        use_linformer=use_linformer
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        # New format (contains complete training information)
        state_dict = checkpoint['model_state_dict']
        
        # Handle DDP-saved models (keys may have 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        
        print(f"\nCheckpoint Information:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'train_loss' in checkpoint:
            print(f"  Training Loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
        if 'val_accuracy' in checkpoint:
            print(f"  Validation Accuracy: {checkpoint['val_accuracy']:.4f} ({checkpoint['val_accuracy']*100:.2f}%)")
    else:
        # Old format (only state_dict)
        model.load_state_dict(checkpoint)
        print("Loaded old format model")
    
    model.eval()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    model.to(device)
    
    # Prepare test text
    print(f"\nTest text: '{text}'")
    tokens = [vocab.char_to_idx[SOS_TOKEN]] + vocab.encode(text) + [vocab.char_to_idx[EOS_TOKEN]]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    else:
        tokens += [vocab.char_to_idx[PAD_TOKEN]] * (MAX_SEQ_LEN - len(tokens))
    
    # Run multiple tests
    print(f"\nRunning 5 test rounds (randomly masking 20% of characters each time)...")
    print("-" * 70)
    n95_list = []
    
    for test_round in range(5):
        masked_tokens = tokens.copy()
        mask_positions = [0] * len(tokens)
        valid_positions = [i for i, t in enumerate(tokens) if
                           t not in [vocab.char_to_idx[SOS_TOKEN], 
                                   vocab.char_to_idx[EOS_TOKEN],
                                   vocab.char_to_idx[PAD_TOKEN]]]
        
        if not valid_positions:
            print(f"Warning: No valid positions to mask")
            continue
        
        num_masks = max(1, int(len(valid_positions) * 0.2))
        mask_indices = random.sample(valid_positions, num_masks)
        
        print(f"\nTest round {test_round + 1}/5:")
        print(f"  Number of mask positions: {num_masks}")
        
        for pos in mask_indices:
            masked_tokens[pos] = vocab.char_to_idx[MASK_TOKEN]
            mask_positions[pos] = 1
        
        input_tensor = torch.tensor([masked_tokens]).to(device)
        labels = torch.tensor(tokens).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            n95, _ = compute_n95(logits[0], labels, torch.tensor(mask_positions))
            n95_mean = n95.float().mean().item()
            n95_list.append(n95_mean)
            
            print(f"  N95 mean: {n95_mean:.2f}")
            
            # Display top-5 candidates
            candidates = get_top_n_candidates(logits[0], torch.tensor(mask_positions), vocab)
            for i, (chars, probs) in enumerate(candidates):
                top5_chars = chars[:5]
                top5_probs = [f"{p:.3f}" for p in probs[:5]]
                print(f"  Mask {i+1}: Top-5 {top5_chars} (Probabilities: {top5_probs})")
    
    # Calculate statistics
    if n95_list:
        n95_mean = np.mean(n95_list)
        n95_std = np.std(n95_list)
        
        print("\n" + "=" * 70)
        print("Test Results Summary:")
        print(f"  N95 mean: {n95_mean:.2f}")
        print(f"  N95 standard deviation: {n95_std:.2f}")
        print("=" * 70)
        
        return n95_mean, n95_std
    else:
        print("\nNo valid test results")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Character Filling Language Model')
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH,
                        help='Model file path (default: {})'.format(MODEL_SAVE_PATH))
    parser.add_argument('--use_linformer', action='store_true',
                        help='Use Linformer attention (must match training configuration)')
    parser.add_argument('--texts', type=str, nargs='+', 
                        default=["獎□先進", "人工智能", "深度学习"],
                        help='List of test texts (default: ["獎□先進", "人工智能", "深度学习"])')
    args = parser.parse_args()
    
    print(f"Model path: {args.model_path}")
    print(f"Attention mechanism: {'Linformer' if args.use_linformer else 'Standard'}")
    print(f"Number of test texts: {len(args.texts)}")
    
    # Test multiple texts
    results = []
    for text in args.texts:
        print("\n" + "=" * 70)
        result = test_model(text, args.model_path, args.use_linformer)
        if result[0] is not None:
            results.append((text, result[0], result[1]))
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print("All Tests Summary:")
        print("=" * 70)
        for text, mean, std in results:
            print(f"'{text}': N95={mean:.2f}±{std:.2f}")