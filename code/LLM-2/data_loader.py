import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from config import *
from tqdm import tqdm


class CharVocab:
    def __init__(self, dictionary_path):
        # Load dictionary and initialize vocabulary
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        
        # Save original vocabulary size first
        original_vocab_size = len(self.char_to_idx)
        
        # Find unoccupied ID for special tokens
        max_id = max(self.char_to_idx.values())
        
        # Assign IDs to special tokens (use existing ID if present, otherwise assign new ID)
        next_id = max_id + 1
        
        # Process each special token
        special_tokens = [PAD_TOKEN, MASK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        special_token_ids = {}
        
        for token in special_tokens:
            if token in self.char_to_idx:
                # If token exists, use original ID
                special_token_ids[token] = self.char_to_idx[token]
                print(f"Special token '{token}' already exists, using original ID: {special_token_ids[token]}")
            else:
                # Otherwise assign new ID
                special_token_ids[token] = next_id
                self.char_to_idx[token] = next_id
                print(f"Special token '{token}' does not exist, assigning new ID: {next_id}")
                next_id += 1
        
        # Build reverse mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Vocabulary size (including special tokens)
        self.vocab_size = len(self.char_to_idx)
        
        print(f"Original vocabulary size: {original_vocab_size}")
        print(f"Vocabulary size after adding special tokens: {self.vocab_size}")

    def encode(self, text):
        # Encode text into token ID sequence
        return [self.char_to_idx.get(char, self.char_to_idx[MASK_TOKEN]) for char in text]

    def decode(self, tokens):
        # Decode token IDs into text
        return ''.join([self.idx_to_char.get(token, MASK_TOKEN) for token in tokens])


class BlankFillingDataset(Dataset):
    def __init__(self, texts, vocab, groups=5, max_len=MAX_SEQ_LEN):
        # Initialize dataset
        self.vocab = vocab
        self.groups = groups
        self.max_len = max_len
        # Store original text segments directly to avoid any heavy preprocessing during initialization
        print("Initializing dataset...")
        self.segments = [seg.strip() for seg in texts if seg.strip()]
        print(f"Dataset initialization completed, loaded {len(self.segments)} original data entries.")

    def __len__(self):
        # Total dataset size is original number of segments * number of sample groups per segment
        return len(self.segments) * self.groups

    def __getitem__(self, idx):
        # --- Perform on-the-fly sample generation here ---

        # 1. Calculate which original segment and processing group it corresponds to based on index idx
        segment_idx = idx // self.groups
        group_idx = idx % self.groups

        # 2. Get original segment and encode it
        segment = self.segments[segment_idx]
        tokens = [self.vocab.char_to_idx[SOS_TOKEN]] + self.vocab.encode(segment) + [
            self.vocab.char_to_idx[EOS_TOKEN]]

        # 3. Pad or truncate to maximum length
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.vocab.char_to_idx[PAD_TOKEN]] * (self.max_len - len(tokens))

        # 4. Execute the same "blank filling" and "add noise" logic as before
        # Exclude mask positions for SOS, EOS, and PAD
        valid_positions = [i for i, t in enumerate(tokens) if t not in [self.vocab.char_to_idx[SOS_TOKEN],
                                                                            self.vocab.char_to_idx[EOS_TOKEN],
                                                                            self.vocab.char_to_idx[PAD_TOKEN]]]

        masked_tokens = tokens.copy()
        mask_positions = [0] * len(tokens)
        mask_labels = [-100] * len(tokens)

        # If there are no valid positions to mask, return directly
        if not valid_positions:
            return {
                "input_ids": torch.tensor(masked_tokens),
                "labels": torch.tensor(mask_labels),
                "mask_positions": torch.tensor(mask_positions)
            }
        
        # Determine mask ratio based on group (based on BERT design concept)
        # BERT original paper uses 15%, research shows 15%-40% works best
        # For groups=5: 15%, 25%, 35%, 45%, 50% (progressive difficulty, avoiding excessive masking)
        mask_ratios = [0.15, 0.25, 0.35, 0.45]
        mask_ratio = mask_ratios[group_idx] if group_idx < len(mask_ratios) else 0.35
        num_masks = max(1, int(len(valid_positions) * mask_ratio))
        mask_indices = random.sample(valid_positions, min(num_masks, len(valid_positions)))

        # Apply masking
        for pos in mask_indices:
            masked_tokens[pos] = self.vocab.char_to_idx[MASK_TOKEN]
            mask_labels[pos] = tokens[pos]
            mask_positions[pos] = 1

        # Add noise - Fix: Use correct ID range
        noise_indices = [i for i in valid_positions if i not in mask_indices]
        if noise_indices:
            noise_count = max(1, int(len(valid_positions) * NOISE_P))
            noise_positions = random.sample(noise_indices, min(noise_count, len(noise_indices)))
            # Fix: Randomly select from all valid character IDs (excluding special tokens)
            special_token_ids = {
                self.vocab.char_to_idx[PAD_TOKEN],
                self.vocab.char_to_idx[MASK_TOKEN],
                self.vocab.char_to_idx[SOS_TOKEN],
                self.vocab.char_to_idx[EOS_TOKEN]
            }
            valid_char_ids = [idx for idx in self.vocab.idx_to_char.keys() if idx not in special_token_ids]
            
            for pos in noise_positions:
                masked_tokens[pos] = random.choice(valid_char_ids)

        # 5. Convert processed data to Tensor and return
        return {
            "input_ids": torch.tensor(masked_tokens),
            "labels": torch.tensor(mask_labels),
            "mask_positions": torch.tensor(mask_positions)
        }