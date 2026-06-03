"""
Combined Version: Generate Fusion Layer Training Data Directly from Small JSON (Confidence-Based Masking Strategy)

Core Concepts:
- OCR_100: Probabilities of the top-100 candidate characters predicted by the OCR model
- LM_100: Probabilities of the SAME 100 candidate characters in the language model (one-to-one correspondence)
- GT_100: One-hot encoding of the ground truth character (among the 100 candidates)

Workflow:
1. Initial Filtering: Remove empty text, length mismatches, and type errors
2. Confidence-Based Masking: Mask the 20% of characters with the lowest OCR confidence (top1 probability)
3. Generate LM Probabilities: Extract LM probabilities for the OCR top-100 candidates only for masked characters
4. Final Filtering: Exclude characters where GT is not in the OCR top-100
5. Output: Character-level training samples (.pt file)

Important Notes:
- LM_100 is NOT the language model's own top-100 predictions
- LM_100 is the probabilities of the OCR's 100 candidates queried from the language model
- All three probability vectors (OCR/LM/GT) correspond to the same 100 candidate characters

🔄 Differences from prepare_fusion_data.py:
- Original Version: Masking based on prediction correctness (mandatory masking for errors + random supplement)
- This Version: Masking based on confidence (mask the 20% with lowest confidence)

Author: Modified based on prepare_fusion_data.py
Date: 2025
"""

import json
import os
import sys
import torch
import numpy as np
import random
import multiprocessing
from tqdm import tqdm
import re

# Add LLM-2 model path
sys.path.insert(0, os.path.abspath('./LLM-2'))
from model import TransformerLM
from config import MASK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


def normalize_symbols(text):
    """Normalize special symbols in text"""
    text = re.sub(r'[【】]', lambda x: '[' if x.group(0) == '【' else ']', text)
    text = re.sub(r'[:：]', ':', text)
    text = re.sub(r'[，,]', ',', text)
    text = text.lower()
    text = text.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
        'abcdefghijklmnopqrstuvwxyz'
    ))
    return text


def load_mappings(map_file, ocr_char_file):
    """Load OCR-to-LM character mapping and OCR character list"""
    with open(map_file, 'r', encoding='utf-8') as f:
        ocr_to_lm = json.load(f)
    with open(ocr_char_file, 'r', encoding='utf-8') as f:
        ocr_chars = [line.strip() for line in f.readlines()]
    return ocr_to_lm, ocr_chars


def extract_decoded_text(decoded_item):
    """Extract decoded text (handles both list and string formats)"""
    if isinstance(decoded_item, list):
        return decoded_item[0] if len(decoded_item) > 0 else ""
    elif isinstance(decoded_item, str):
        return decoded_item
    else:
        raise ValueError(f"Unexpected type for decoded_item: {type(decoded_item)}")


class BidirectionalProbabilityGenerator:
    """
    Bidirectional Language Model Probability Generator
    
    Based on LLM-2's cloze-style Transformer model
    Predicts probability distribution for each character by masking different positions
    and leveraging full contextual information
    """
    
    def __init__(self, model_path, vocab_path, device=None):
        """
        Initialize bidirectional language model
        
        Args:
            model_path: Path to LLM-2 model weights (.pth)
            vocab_path: Path to character-to-index mapping file (char_to_idx.json)
            device: Running device (auto-detected by default)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        self.char_to_idx = self._load_vocab(vocab_path)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # IDs of special tokens
        self.mask_token_id = self.char_to_idx.get(MASK_TOKEN, 1)
        self.pad_token_id = self.char_to_idx.get(PAD_TOKEN, 0)
        self.sos_token_id = self.char_to_idx.get(SOS_TOKEN, 2)
        self.eos_token_id = self.char_to_idx.get(EOS_TOKEN, 3)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_vocab(self, vocab_path):
        """Load vocabulary and add special tokens"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            char_to_idx = json.load(f)
        
        # Calculate next available ID (start from max ID + 1)
        max_id = max(char_to_idx.values())
        next_id = max_id + 1
        
        # Check and add special tokens in training order
        special_tokens = [PAD_TOKEN, MASK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        
        for token in special_tokens:
            if token not in char_to_idx:
                char_to_idx[token] = next_id
                next_id += 1
        
        return char_to_idx
    
    def _load_model(self, model_path):
        """Load TransformerLM model"""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            embed_dim=512,
            hidden_dim=3072,
            num_layers=24,
            num_heads=8,
            dropout=0.1,
            use_linformer=True  # Use Linformer attention
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _encode_text(self, text):
        """Encode text into index sequence"""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Replace unknown characters with mask token
                encoded.append(self.mask_token_id)
        return encoded
    
    def get_masked_positions_probs(self, decoded_text, mask_positions, max_seq_len=128):
        """
        Generate context-aware probability distributions for specified masked positions
        
        Args:
            decoded_text: Complete decoded text (str)
            mask_positions: List of positions to mask (List[int])
            max_seq_len: Maximum sequence length supported by the model (default: 128)
        
        Returns:
            probs_dict: Dict[int, np.ndarray], key is position, value is [vocab_size] probability distribution
        """
        if not decoded_text or not mask_positions:
            return {}
        
        seq_len = len(decoded_text)
        probs_dict = {}
        
        # Process each masked position
        for pos in mask_positions:
            if pos >= seq_len:
                continue
            
            # Select inference strategy based on sequence length
            if seq_len <= max_seq_len:
                # Sequence within limit, infer directly
                prob = self._get_single_position_probs_internal(decoded_text, pos)
            else:
                # Sequence exceeds limit, use sliding window
                if pos < max_seq_len:
                    # First max_seq_len positions: use [0:max_seq_len] window
                    window_text = decoded_text[:max_seq_len]
                    position_in_window = pos
                else:
                    # Subsequent positions: use sliding window
                    window_start = pos - max_seq_len + 1
                    window_end = pos + 1
                    window_text = decoded_text[window_start:window_end]
                    position_in_window = len(window_text) - 1
                
                prob = self._get_single_position_probs_internal(window_text, position_in_window)
            
            if prob is not None:
                probs_dict[pos] = prob
            else:
                # Use uniform distribution as fallback if retrieval fails
                uniform_prob = np.ones(self.vocab_size) / self.vocab_size
                probs_dict[pos] = uniform_prob
        
        return probs_dict
    
    def _get_single_position_probs_internal(self, decoded_text, position):
        """
        Generate probability distribution for a single position (internal helper method)
        
        Args:
            decoded_text: Decoded text (caller ensures valid length)
            position: Target position (0-based)
        
        Returns:
            probs: np.ndarray [vocab_size]
        """
        if position >= len(decoded_text):
            return None
        
        # Create masked input
        masked_chars = list(decoded_text)
        masked_chars[position] = MASK_TOKEN
        masked_text = ''.join(masked_chars)
        
        # Encode and infer
        input_ids = self._encode_text(masked_text)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)  # [1, seq_len, vocab_size]
            probs = torch.softmax(logits[0, position, :], dim=-1)
        
        return probs.cpu().numpy()


def apply_confidence_masking(sample_topk_probs, mask_ratio=0.2, seed=None):
    """
    🆕 Apply confidence-based masking strategy:
    Mask the mask_ratio (20%) of characters with the lowest OCR confidence (top1 probability)
    
    Args:
        sample_topk_probs: OCR top-100 probabilities [seq_len, 100]
        mask_ratio: Target masking rate (default: 20%)
        seed: Random seed (optional, not used here, kept for interface consistency)
    
    Returns:
        mask_positions: List of masked positions (List[int]), sorted by position
    """
    seq_len = len(sample_topk_probs)
    target_mask_count = max(1, int(seq_len * mask_ratio))  # Mask at least 1 character
    
    # Calculate OCR confidence (top1 probability) for each position
    confidence_scores = []
    for i, probs in enumerate(sample_topk_probs):
        if len(probs) > 0:
            top1_prob = probs[0]  # Probability of top1 candidate
        else:
            top1_prob = 0.0  # Zero confidence if no candidates
        
        confidence_scores.append((i, top1_prob))
    
    # Sort by confidence in ascending order (lowest to highest)
    confidence_scores.sort(key=lambda x: x[1])
    
    # Select target_mask_count positions with lowest confidence
    mask_positions = [pos for pos, _ in confidence_scores[:target_mask_count]]
    
    # Return sorted by position
    mask_positions.sort()
    
    return mask_positions


def generate_lm_probs_for_masks(decoded_text, mask_positions, bidirectional_lm, 
                                 ocr_to_lm, topk_indices):
    """
    Generate LM probability distributions for masked positions (one-to-one with OCR top-100 candidates)
    
    Important Notes:
    - LM_100 is NOT the language model's own top-100 predictions
    - LM_100 is the probabilities of the OCR's top-100 candidates queried from the language model
    - Maintains the same candidate list as OCR_100, only differs in probability source
    
    Args:
        decoded_text: Decoded text (str)
        mask_positions: List of masked positions (List[int])
        bidirectional_lm: Bidirectional LM instance
        ocr_to_lm: OCR index to LM index mapping (dict)
        topk_indices: OCR top-100 indices [seq_len, 100]
    
    Returns:
        lm_probs_dict: Dict[int, List[float]], key is position, value is LM probabilities (one-to-one with OCR top-100)
    """
    # 1. Batch retrieve complete LM probability distributions for masked positions
    all_probs_dict = bidirectional_lm.get_masked_positions_probs(
        decoded_text, 
        mask_positions
    )  # Dict[int, np.ndarray[vocab_size]], contains probabilities for all characters in LM vocabulary
    
    # 2. For each masked position, extract LM probabilities corresponding to OCR top-100 candidates
    lm_probs_dict = {}
    
    for pos in mask_positions:
        if pos not in all_probs_dict or pos >= len(topk_indices):
            continue
        
        probs_full = all_probs_dict[pos]  # Complete LM probability distribution [vocab_size]
        
        # Extract LM probabilities for OCR top-100 candidates (one-to-one correspondence)
        lm_probs_selected = []
        for ocr_idx in topk_indices[pos]:  # Iterate through OCR's 100 candidate indices
            # Map OCR index to LM vocabulary index
            lm_idx = ocr_to_lm.get(str(ocr_idx), -1)
            
            if lm_idx != -1 and lm_idx < len(probs_full):
                # Extract probability of this character from LM's complete distribution
                lm_probs_selected.append(float(probs_full[lm_idx]))
            else:
                # OCR character not in LM vocabulary or index out of bounds, use minimal probability
                lm_probs_selected.append(1e-10)
        
        # Ensure length is 100
        assert len(lm_probs_selected) == len(topk_indices[pos]), \
            f"LM probability length {len(lm_probs_selected)} should equal OCR top-100 length {len(topk_indices[pos])}"
        
        lm_probs_dict[pos] = lm_probs_selected
    
    return lm_probs_dict


def generate_character_samples(norm_decoded_text, norm_true_text, 
                                mask_positions, sample_topk_indices, 
                                sample_topk_probs, lm_probs_dict, 
                                ocr_chars):
    """
    Generate character-level training samples for masked positions
    
    Args:
        norm_decoded_text: Normalized decoded text
        norm_true_text: Normalized ground truth text
        mask_positions: List of masked positions
        sample_topk_indices: OCR top-100 indices [seq_len, 100]
        sample_topk_probs: OCR top-100 probabilities [seq_len, 100]
        lm_probs_dict: LM probability dictionary Dict[int, List[float]]
        ocr_chars: OCR character list
    
    Returns:
        char_samples: List[Dict], each element contains ocr_c100, lm_c100, gt_c100, topk_indices, is_correct
        stats: Statistical information
    """
    char_samples = []
    stats = {
        'total_masked': len(mask_positions),
        'filtered_gt_not_in_top100': 0,
        'final_correct': 0,
        'final_incorrect': 0
    }
    
    for pos in mask_positions:
        # Check if LM probabilities exist for the position
        if pos not in lm_probs_dict:
            continue
        
        # Get ground truth character
        if pos >= len(norm_true_text):
            continue
        
        gt_char = norm_true_text[pos]
        
        # Get OCR top-100 indices and probabilities
        topk_idx = sample_topk_indices[pos]
        topk_prob = sample_topk_probs[pos]
        
        # Check if GT is in OCR top-100
        gt_in_top100 = False
        gt_position_in_top100 = -1
        
        for i, ocr_idx in enumerate(topk_idx):
            # ocr_idx is 1-based, need to subtract 1
            if 0 < ocr_idx <= len(ocr_chars):
                ocr_char = ocr_chars[ocr_idx - 1]
                if ocr_char == gt_char:
                    gt_in_top100 = True
                    gt_position_in_top100 = i
                    break
        
        # Stage 4 Filtering: Skip if GT not in top-100
        if not gt_in_top100:
            stats['filtered_gt_not_in_top100'] += 1
            continue
        
        # Construct one-hot encoding of GT (among top-100)
        gt_c100 = [0.0] * len(topk_idx)
        gt_c100[gt_position_in_top100] = 1.0
        
        # Get OCR and LM probabilities
        ocr_c100 = topk_prob
        lm_c100 = lm_probs_dict[pos]
        
        # Determine if OCR prediction is correct (for statistics)
        ocr_pred_char = ocr_chars[topk_idx[0] - 1] if 0 < topk_idx[0] <= len(ocr_chars) else ''
        is_correct = (ocr_pred_char == gt_char)
        
        # Save character sample
        char_samples.append({
            'ocr_c100': ocr_c100,
            'lm_c100': lm_c100,
            'gt_c100': gt_c100,
            'topk_indices': topk_idx,
            'is_correct': is_correct
        })
        
        # Update statistics
        if is_correct:
            stats['final_correct'] += 1
        else:
            stats['final_incorrect'] += 1
    
    return char_samples, stats


def process_single_file_worker(args):
    """
    Worker function to process a single JSON file
    
    Args:
        args: Tuple (json_file_path, gpu_id, map_file, ocr_char_file, 
                    bidirectional_lm_path, vocab_path, mask_ratio, seed)
    
    Returns:
        char_samples: List of character-level training samples
        stats: Statistical information dictionary
    """
    (json_file_path, gpu_id, map_file, ocr_char_file, 
     bidirectional_lm_path, vocab_path, mask_ratio, seed) = args
    
    # ⚠️ Important: Must set CUDA_VISIBLE_DEVICES before importing torch
    # But since torch is already imported at the top, we specify device directly
    # Each process uses the assigned GPU (cuda:0, cuda:1, ...)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Load mappings and character list
    ocr_to_lm, ocr_chars = load_mappings(map_file, ocr_char_file)
    
    # Load bidirectional language model (specify GPU to use)
    bidirectional_lm = BidirectionalProbabilityGenerator(
        model_path=bidirectional_lm_path,
        vocab_path=vocab_path,
        device=device  # Explicitly specify device
    )
    
    # Initialize statistics
    all_char_samples = []
    stats = {
        # Stage 1 Filtering
        'stage1_total_samples': 0,
        'stage1_filtered_empty': 0,
        'stage1_filtered_length_mismatch': 0,
        'stage1_filtered_type_error': 0,
        
        # Stage 2 Masking (confidence-based)
        'stage2_passed_samples': 0,
        'stage2_total_chars': 0,
        'stage2_masked_chars': 0,
        
        # Stage 3 LM Probability Generation (successful)
        'stage3_lm_generated': 0,
        
        # Stage 4 GT Filtering
        'stage4_filtered_gt_not_in_top100': 0,
        'stage4_final_chars': 0,
        'stage4_final_correct': 0,
        'stage4_final_incorrect': 0
    }
    
    # Read JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return all_char_samples, stats
    
    # Extract batch data
    topk_probs = data.get('topk_probs', [])
    topk_indices = data.get('topk_indices', [])
    labels = data.get('labels', [])
    decoded_texts = data.get('decoded_texts', [])
    
    # Process each sample
    for sample_idx in range(len(decoded_texts)):
        stats['stage1_total_samples'] += 1
        
        # ========== Stage 1 Filtering: Empty text, type errors, length mismatches ==========
        
        # Extract ground truth and decoded text
        try:
            true_text = labels[sample_idx][0] if isinstance(labels[sample_idx], list) else labels[sample_idx]
            decoded_item = decoded_texts[sample_idx]
            decoded_text = extract_decoded_text(decoded_item)
            
            # Type check
            if not isinstance(decoded_text, str) or not isinstance(true_text, str):
                stats['stage1_filtered_type_error'] += 1
                continue
            
            # Normalize text
            norm_true_text = normalize_symbols(true_text)
            norm_decoded_text = normalize_symbols(decoded_text)
            
            true_len = len(norm_true_text)
            decoded_len = len(norm_decoded_text)
            
            # Check if decoded text is empty
            if decoded_len == 0:
                stats['stage1_filtered_empty'] += 1
                continue
            
            # Check if sample length is less than 5
            if true_len < 5 or decoded_len < 5:
                stats['stage1_filtered_empty'] += 1  # Categorize into empty statistics
                continue
            
            # Get top-100 data for the sample
            sample_topk_indices = topk_indices[sample_idx]
            sample_topk_probs = topk_probs[sample_idx]
            
            # Check length consistency
            if (len(sample_topk_indices) != true_len or 
                len(sample_topk_indices) != decoded_len):
                stats['stage1_filtered_length_mismatch'] += 1
                continue
                
        except (IndexError, KeyError, TypeError, ValueError) as e:
            stats['stage1_filtered_type_error'] += 1
            continue
        
        # Passed Stage 1 Filtering
        stats['stage2_passed_samples'] += 1
        stats['stage2_total_chars'] += len(norm_decoded_text)
        
        # Stage 2: Confidence-based masking
        mask_positions = apply_confidence_masking(
            sample_topk_probs, 
            mask_ratio=mask_ratio, 
            seed=seed
        )
        
        stats['stage2_masked_chars'] += len(mask_positions)
        
        # Skip if no positions are masked
        if not mask_positions:
            continue
        
        # Stage 3: Generate LM probabilities for masked positions
        lm_probs_dict = generate_lm_probs_for_masks(
            norm_decoded_text, 
            mask_positions, 
            bidirectional_lm, 
            ocr_to_lm, 
            sample_topk_indices
        )
        
        stats['stage3_lm_generated'] += len(lm_probs_dict)
        
        # Skip if no LM probabilities are successfully generated
        if not lm_probs_dict:
            continue
        
        # Stage 4: Generate character samples and apply GT filtering
        char_samples, sample_stats = generate_character_samples(
            norm_decoded_text, 
            norm_true_text, 
            mask_positions, 
            sample_topk_indices, 
            sample_topk_probs, 
            lm_probs_dict, 
            ocr_chars
        )
        
        # Update statistics
        stats['stage4_filtered_gt_not_in_top100'] += sample_stats['filtered_gt_not_in_top100']
        stats['stage4_final_chars'] += len(char_samples)
        stats['stage4_final_correct'] += sample_stats['final_correct']
        stats['stage4_final_incorrect'] += sample_stats['final_incorrect']
        
        # Collect samples
        all_char_samples.extend(char_samples)
    
    return all_char_samples, stats


def prepare_fusion_data(json_folder, map_file, ocr_char_file, 
                       bidirectional_lm_path, vocab_path, 
                       output_file, mask_ratio=0.2, seed=42, 
                       gpu_ids=None, num_workers_per_gpu=1):
    """
    Prepare fusion layer training data (multi-process processing, confidence-based masking strategy)
    
    Args:
        json_folder: Path to JSON folder
        map_file: Path to OCR-to-LM mapping file
        ocr_char_file: Path to OCR character list file
        bidirectional_lm_path: Path to bidirectional LM model
        vocab_path: Path to LM vocabulary file
        output_file: Path to output file (.pt)
        mask_ratio: Target masking rate (default: 20%)
        seed: Random seed
        gpu_ids: List of GPU IDs to use (None by default, uses all available GPUs)
        num_workers_per_gpu: Number of workers per GPU (default: 1, recommended to avoid out-of-memory)
    """
    # Detect number of GPUs
    total_gpus = torch.cuda.device_count()
    if total_gpus == 0:
        raise SystemExit("Error: No CUDA GPUs found. This script requires at least one GPU for distributed processing.")
    
    # Determine GPUs to use
    if gpu_ids is None:
        gpu_ids = list(range(total_gpus))
    else:
        # Validate specified GPU IDs
        for gid in gpu_ids:
            if gid >= total_gpus:
                raise ValueError(f"GPU {gid} does not exist! The system only has {total_gpus} GPUs (0-{total_gpus-1})")
    
    num_gpus = len(gpu_ids)
    print(f"Detected {total_gpus} GPUs, using GPUs: {gpu_ids}")
    print(f"Number of workers per GPU: {num_workers_per_gpu}")
    print(f"Total number of workers: {num_gpus * num_workers_per_gpu}")
    
    # Get all JSON files
    json_files = [
        os.path.join(json_folder, f) 
        for f in os.listdir(json_folder) 
        if f.endswith('.json')
    ]
    print(f"Found {len(json_files)} JSON files")
    
    # Construct task list (cycle GPU assignment)
    tasks = []
    for i, json_file in enumerate(json_files):
        # Cycle through GPU IDs in gpu_ids list
        gpu_id = gpu_ids[i % num_gpus]
        tasks.append((json_file, gpu_id, map_file, ocr_char_file, 
                     bidirectional_lm_path, vocab_path, mask_ratio, seed))
    
    # Initialize total statistics
    all_char_samples = []
    total_stats = {
        # Stage 1 Filtering
        'stage1_total_samples': 0,
        'stage1_filtered_empty': 0,
        'stage1_filtered_length_mismatch': 0,
        'stage1_filtered_type_error': 0,
        
        # Stage 2 Masking (confidence-based)
        'stage2_passed_samples': 0,
        'stage2_total_chars': 0,
        'stage2_masked_chars': 0,
        
        # Stage 3 LM Probability Generation
        'stage3_lm_generated': 0,
        
        # Stage 4 GT Filtering
        'stage4_filtered_gt_not_in_top100': 0,
        'stage4_final_chars': 0,
        'stage4_final_correct': 0,
        'stage4_final_incorrect': 0
    }
    
    # Multi-process processing (limit concurrency to avoid out-of-memory)
    total_workers = num_gpus * num_workers_per_gpu
    with multiprocessing.get_context("spawn").Pool(processes=total_workers) as pool:
        results_iterator = pool.imap_unordered(process_single_file_worker, tasks)
        
        # Show progress bar
        pbar = tqdm(results_iterator, total=len(tasks), desc="Processing JSON files")
        
        for result_from_worker in pbar:
            char_samples, stats = result_from_worker
            
            # Aggregate samples
            all_char_samples.extend(char_samples)
            
            # Aggregate statistics
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 Data Preparation Statistics (Confidence-Based Masking Strategy)")
    print("="*80)
    
    print(f"\n[Stage 1] Initial Filtering")
    print(f"  Total samples:           {total_stats['stage1_total_samples']}")
    print(f"  ├─ Filtered (empty):     {total_stats['stage1_filtered_empty']}")
    print(f"  ├─ Filtered (length mismatch): {total_stats['stage1_filtered_length_mismatch']}")
    print(f"  ├─ Filtered (type error):     {total_stats['stage1_filtered_type_error']}")
    print(f"  └─ Passed:               {total_stats['stage2_passed_samples']}")
    
    print(f"\n[Stage 2] Confidence-Based Masking")
    print(f"  Number of samples:       {total_stats['stage2_passed_samples']}")
    print(f"  Total characters:        {total_stats['stage2_total_chars']}")
    print(f"  Masked characters:       {total_stats['stage2_masked_chars']}")
    if total_stats['stage2_total_chars'] > 0:
        mask_rate = total_stats['stage2_masked_chars'] / total_stats['stage2_total_chars'] * 100
        print(f"  Actual masking rate:     {mask_rate:.2f}%")
    print(f"  📌 Strategy Note: Mask the {mask_ratio*100:.0f}% of characters with the lowest OCR confidence (top1 probability)")
    
    print(f"\n[Stage 3] LM Probability Generation")
    print(f"  Characters with successful LM probability generation: {total_stats['stage3_lm_generated']}")
    
    print(f"\n[Stage 4] GT Filtering & Final Samples")
    print(f"  Filtered (GT not in top-100): {total_stats['stage4_filtered_gt_not_in_top100']}")
    print(f"  Final character samples:      {total_stats['stage4_final_chars']}")
    print(f"  ├─ OCR predicted correctly:   {total_stats['stage4_final_correct']}")
    print(f"  └─ OCR predicted incorrectly: {total_stats['stage4_final_incorrect']}")
    
    if total_stats['stage4_final_chars'] > 0:
        correct_rate = total_stats['stage4_final_correct'] / total_stats['stage4_final_chars'] * 100
        print(f"  OCR accuracy:                {correct_rate:.2f}%")
    
    print("="*80)
    
    # Convert to tensors and save
    if len(all_char_samples) == 0:
        print("\n⚠️ Warning: No valid samples generated!")
        return
    
    print(f"\nConverting to tensors and saving to {output_file}...")
    
    # Extract data
    ocr_list = [s['ocr_c100'] for s in all_char_samples]
    lm_list = [s['lm_c100'] for s in all_char_samples]
    gt_list = [s['gt_c100'] for s in all_char_samples]
    topk_indices_list = [s['topk_indices'] for s in all_char_samples]
    is_correct_list = [s['is_correct'] for s in all_char_samples]
    
    # Convert to tensors
    ocr_tensor = torch.tensor(ocr_list, dtype=torch.float32)
    lm_tensor = torch.tensor(lm_list, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_list, dtype=torch.float32)
    topk_indices_tensor = torch.tensor(topk_indices_list, dtype=torch.int32)
    is_correct_tensor = torch.tensor(is_correct_list, dtype=torch.bool)
    
    # Save
    torch.save({
        'ocr_c100': ocr_tensor,
        'lm_c100': lm_tensor,
        'gt_c100': gt_tensor,
        'topk_indices': topk_indices_tensor,
        'is_correct': is_correct_tensor,
        'stats': total_stats,
        'mask_strategy': 'confidence_based',  # Mark masking strategy
        'mask_ratio': mask_ratio
    }, output_file)
    
    print(f"✅ Saved successfully!")
    print(f"\nData shapes:")
    print(f"  ├─ OCR:          {ocr_tensor.shape}")
    print(f"  ├─ LM:           {lm_tensor.shape}")
    print(f"  ├─ GT:           {gt_tensor.shape}")
    print(f"  ├─ topk_indices: {topk_indices_tensor.shape}")
    print(f"  └─ is_correct:   {is_correct_tensor.shape}")

if __name__ == "__main__":
    """
    Usage Example
    """
    
    # ========== Configuration Section ==========
    # Specify GPUs to use (if not specified, uses all GPUs by default)
    # Examples:
    #   gpu_ids = None         # Use all GPUs
    #   gpu_ids = [0]          # Use only GPU 0
    #   gpu_ids = [0, 1]       # Use GPUs 0 and 1
    #   gpu_ids = [1, 2, 3]    # Use GPUs 1, 2, 3
    
    gpu_ids = [0, 1, 2, 3, 4, 5]  # Modify here to specify GPUs
    num_workers_per_gpu = 1  # Number of workers per GPU (recommended: 1 to avoid out-of-memory)
    
    prepare_fusion_data(
        json_folder='./gky/nrtr_image_hwtrain_top100_3',          # JSON folder (small JSON files)
        map_file='map.json',                                      # OCR-to-LM mapping file
        ocr_char_file='ppocr_keys_v1.txt',                       # OCR character list file
        bidirectional_lm_path='./language_model/checkpoint_epoch_14.pth',       # Bidirectional LM model path
        vocab_path='./data/char_to_idx.json',                    # LM vocabulary path
        output_file='./new_pt_data/nrtr/train/nrtr_image_hwtrain_confidence_3.pt',        # Output file (new name)
        mask_ratio=0.2,                                           # Masking rate: 20%
        seed=42,                                                  # Random seed
        gpu_ids=gpu_ids,                                          # Specify GPUs to use
        num_workers_per_gpu=num_workers_per_gpu                  # Number of workers per GPU
    )
    
    print("\n✅ Fusion layer training data preparation completed!")
    print("📌 Notes:") 
    print("   - Masking Strategy: Based on OCR confidence (top1 probability)")
    print("   - Masked Target: 20% of characters with the lowest confidence")
    print("   - Comparison with Original Version: Original uses prediction correctness, this version uses confidence")
    print("   - Saved File: fusion_training_data_confidence.pt")