#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-line Level Fusion Model Testing Script

Testing Workflow:
1. Read JSON files (containing OCR top-100 data and mini-line segmentation information)
2. At major line level: Mask the characters with the lowest 20% of OCR top1 probabilities
3. At major line level: Use bidirectional language model to predict masked characters and generate lm_100
4. At major line level: Combine ocr_100 and lm_100, use fusion model to predict final characters
5. At major line level: Replace masked characters to get the predicted text of the major line
6. Segment the major line into mini-lines according to preds_per_line_len and labels_per_line_len in JSON
7. Normalize the predicted text and true labels of each mini-line (full-width to half-width, 
   traditional Chinese to simplified Chinese, case conversion, space removal, punctuation conversion)
8. Calculate ACC, AR, CR metrics by comparing each mini-line's predicted text with true labels

Metric Explanations:
- ACC: Mini-line Level Accuracy (Number of fully correct mini-lines / Total mini-lines)
- AR (Acceptance Rate): AR = (N_t - D_e - S_e - I_e) / N_t
- CR (Correction Rate): CR = (N_t - D_e - S_e) / N_t
  Where:
    N_t: Total number of characters
    D_e: Number of deletion errors
    S_e: Number of substitution errors
    I_e: Number of insertion errors

Text Normalization:
- Full-width to half-width (NFKC normalization)
- Traditional Chinese to simplified Chinese (requires opencc library)
- Uppercase to lowercase
- Remove all spaces
- Convert Chinese punctuation to English punctuation
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import glob
import unicodedata
import re
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Try to import opencc (for traditional to simplified Chinese conversion)
try:
    import opencc
    cc = opencc.OpenCC('t2s')  # Traditional to simplified Chinese
except ImportError:
    cc = None
    print("⚠️ Warning: opencc not installed, traditional to simplified Chinese conversion unavailable")

# Add paths
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from fusion_model_clean import CrossAttentionFusion
from config_fusion import *

# Import bidirectional language model
# Dynamically import since module names cannot start with numbers
import importlib.util

def load_bidirectional_lm_module():
    """Dynamically load bidirectional language model module"""
    module_path = os.path.join(__dir__, '2_project_svtr_bidirectional.py')
    spec = importlib.util.spec_from_file_location("bidirectional_lm_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    bidirectional_lm_module = load_bidirectional_lm_module()
    BidirectionalProbabilityGenerator = bidirectional_lm_module.BidirectionalProbabilityGenerator
    load_mappings = bidirectional_lm_module.load_mappings
except Exception as e:
    # If import fails, try importing from prepare_fusion_data
    try:
        from prepare_fusion_data import BidirectionalProbabilityGenerator
    except ImportError:
        print(f"❌ Failed to import BidirectionalProbabilityGenerator: {e}")
        raise
    
    # Manually implement load_mappings
    def load_mappings(map_file, ocr_char_file):
        with open(map_file, 'r', encoding='utf-8') as f:
            ocr_to_lm = json.load(f)
        with open(ocr_char_file, 'r', encoding='utf-8') as f:
            ocr_chars = [line.strip() for line in f.readlines()]
        return ocr_to_lm, ocr_chars


def load_json_data(json_file):
    """Load data from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_mask_positions(topk_probs, mask_ratio=0.2):
    """
    Find positions of characters to mask (characters with the lowest mask_ratio of OCR top1 probabilities)
    
    Args:
        topk_probs: [seq_len, 100] OCR top-100 probability list
        mask_ratio: Mask ratio (default: 0.2, i.e., lowest 20%)
    
    Returns:
        mask_positions: List of indices of positions to mask (sorted by probability from lowest to highest)
    """
    seq_len = len(topk_probs)
    if seq_len == 0:
        return []
    
    # Calculate OCR top1 probability for each position
    top1_probs = [probs[0] if len(probs) > 0 else 0.0 for probs in topk_probs]
    
    # Sort by probability and find positions with lowest mask_ratio
    positions_with_probs = [(i, prob) for i, prob in enumerate(top1_probs)]
    positions_with_probs.sort(key=lambda x: x[1])  # Sort by probability from lowest to highest
    
    # Select positions with lowest mask_ratio
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_positions = [pos for pos, _ in positions_with_probs[:num_mask]]
    
    return sorted(mask_positions)


def generate_lm_probs_for_masks(decoded_text, mask_positions, bidirectional_lm, ocr_to_lm, topk_indices):
    """
    Generate LM top-100 probabilities for masked positions
    
    Args:
        decoded_text: Decoded text (string)
        mask_positions: List of positions to mask
        bidirectional_lm: Bidirectional language model instance
        ocr_to_lm: OCR index to LM index mapping
        topk_indices: OCR top-100 index list [seq_len, 100]
    
    Returns:
        lm_c100_dict: {position: [100 probability values]} dictionary (unnormalized, will be normalized in fuse_and_replace)
    """
    if not mask_positions:
        return {}
    
    # Get complete probability distribution for all positions
    all_probs = bidirectional_lm.get_batch_contextual_probs(decoded_text)  # [seq_len, vocab_size]
    
    lm_c100_dict = {}
    
    for pos in mask_positions:
        if pos >= len(all_probs) or pos >= len(topk_indices):
            # Mask position out of range, skip
            continue
        
        probs_full = all_probs[pos]
        ocr_indices = topk_indices[pos]
        
        # Extract LM probabilities corresponding to OCR top-100 candidates
        lm_probs_selected = []
        for ocr_idx in ocr_indices:
            # Map to LM vocabulary
            lm_idx = ocr_to_lm.get(str(ocr_idx), -1)
            
            if lm_idx != -1 and lm_idx < len(probs_full):
                lm_probs_selected.append(float(probs_full[lm_idx]))
            else:
                # OCR character not in LM vocabulary, use small probability
                lm_probs_selected.append(1e-10)
        
        # Note: No normalization here, normalization will be performed uniformly in fuse_and_replace
        # This ensures consistency with the normalization method during training
        lm_c100_dict[pos] = lm_probs_selected
    
    return lm_c100_dict


def fuse_and_replace(decoded_text, mask_positions, topk_probs, topk_indices, 
                     lm_c100_dict, fusion_model, ocr_chars, device):
    """
    Use fusion model to predict masked characters and replace in original text
    
    Args:
        decoded_text: Original decoded text
        mask_positions: List of mask positions
        topk_probs: OCR top-100 probability list [seq_len, 100]
        topk_indices: OCR top-100 index list [seq_len, 100]
        lm_c100_dict: {position: [100 LM probabilities]} dictionary
        fusion_model: Fusion model
        ocr_chars: OCR character list
        device: Device
    
    Returns:
        new_text: New text after replacement (list, each element is a character)
        predictions: {position: predicted_char} dictionary
    """
    new_text = list(decoded_text)
    predictions = {}
    
    if not mask_positions:
        return new_text, predictions
    
    # Batch process mask positions
    batch_ocr = []
    batch_lm = []
    batch_positions = []
    
    for pos in mask_positions:
        if pos not in lm_c100_dict:
            continue
        
        # Prepare OCR and LM probabilities (raw probabilities, unnormalized)
        ocr_probs = np.array(topk_probs[pos], dtype=np.float32)
        lm_probs = np.array(lm_c100_dict[pos], dtype=np.float32)
        
        # Normalization (ensure sum of probabilities is 1.0)
        # Use exactly the same method as in normalize_data.py and training pipeline
        # For 1D arrays, sum() is equivalent to sum(dim=-1), but we use the same method for clarity
        ocr_sum = ocr_probs.sum()
        lm_sum = lm_probs.sum()
        
        # Normalization: Ensure sum of probabilities is 1.0 (consistent with normalize_data.py)
        # Formula: prob_normalized = prob / (sum(prob) + 1e-10)
        # Exactly the same as the normalization method during training
        ocr_probs_normalized = ocr_probs / (ocr_sum + 1e-10)
        lm_probs_normalized = lm_probs / (lm_sum + 1e-10)
        
        # Convert to torch tensor [1, 1, 100]
        # Note: Fusion model expects input shape [batch, 1, 100]
        ocr_tensor = torch.tensor(ocr_probs_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 100]
        lm_tensor = torch.tensor(lm_probs_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)   # [1, 1, 100]
        
        batch_ocr.append(ocr_tensor)
        batch_lm.append(lm_tensor)
        batch_positions.append(pos)
    
    if not batch_ocr:
        return new_text, predictions
    
    # Combine into batch
    batch_ocr = torch.cat(batch_ocr, dim=0).to(device)  # [batch, 1, 100]
    batch_lm = torch.cat(batch_lm, dim=0).to(device)     # [batch, 1, 100]
    
    # Fusion model inference
    with torch.no_grad():
        fusion_logits = fusion_model(batch_ocr, batch_lm)  # [batch, 100]
        fusion_preds = fusion_logits.argmax(dim=-1)       # [batch] - position in top-100
    
    # Replace characters
    for i, pos in enumerate(batch_positions):
        pred_idx_in_top100 = fusion_preds[i].item()
        
        # Check index range
        if pred_idx_in_top100 >= len(topk_indices[pos]):
            # Predicted index out of range, skip this position
            continue
        
        ocr_idx = topk_indices[pos][pred_idx_in_top100]  # OCR index (1-based)
        
        # Convert OCR index to character (index is 1-based, character list is 0-based)
        if 1 <= ocr_idx <= len(ocr_chars):
            predicted_char = ocr_chars[ocr_idx - 1]
            new_text[pos] = predicted_char
            predictions[pos] = predicted_char
        # If index is invalid, no replacement (keep original character)
    
    return new_text, predictions


def _unify_evaluation_text(text):
    """
    Unified normalization processing for evaluation text
    
    Args:
        text: Text string to be normalized
    
    Returns:
        Normalized text string
    """
    # 1. Full-width to half-width
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Traditional Chinese to simplified Chinese (requires opencc installation)
    if cc:
        text = cc.convert(text)
    # else: Warning already printed during import, no duplicate print here
    
    # 3. Uppercase to lowercase
    text = text.lower()
    
    # 4. Remove all spaces
    text = re.sub(r'\s+', '', text)
    
    # 5. Other symbol conversions
    punctuation_map = {
        '【': '[', '】': ']',
        '：': ':', '，': ',', '；': ';', '！': '!', '？': '?',
        '（': '(', '）': ')', '《': '<', '》': '>', '＂': '"', '＇': "'"
    }
    for cn_punct, en_punct in punctuation_map.items():
        text = text.replace(cn_punct, en_punct)
    
    return text

def calculate_edit_distance(pred, target):
        """Calculate edit operations: number of Deletion (De), Substitution (Se), Insertion (Ie) errors"""
        m, n = len(pred), len(target)
        
        # Use dynamic programming matrix to analyze edit operations (from pred->target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialization
        for i in range(m + 1):
            dp[i][0] = i  # Deletion errors needed to convert pred[:i] to empty string
        for j in range(n + 1):
            dp[0][j] = j  # Insertion errors needed to convert empty string to target[:j]
        
        # Fill DP matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No error, inherit previous state
                else:
                    dp[i][j] = min(dp[i-1][j] + 1,    # Deletion: remove pred[i-1]
                                dp[i][j-1] + 1,     # Insertion: add target[j-1] to pred
                                dp[i-1][j-1] + 1)   # Substitution: replace pred[i-1] with target[j-1]
        
        # Backtrack to analyze specific operations (from target->pred)
        de, se, ie = 0, 0, 0
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred[i-1] == target[j-1]:
                # Character match, no operation
                i -= 1
                j -= 1
            else:
                if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                    se += 1  # Substitution error
                    i -= 1
                    j -= 1
                elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                    # From target to pred requires insertion → Insertion error (Ie)
                    ie += 1  
                    i -= 1
                else:
                    # From target to pred requires deletion → Deletion error (De)  
                    de += 1  
                    j -= 1
        
        return dp[m][n], se, de, ie


def split_into_mini_lines(text, line_lengths):
    """
    Split major line text into mini-lines according to line lengths
    
    Args:
        text: Major line text (string)
        line_lengths: List of character lengths for each mini-line, e.g., [9, 10]
    
    Returns:
        mini_lines: List of mini-line texts, e.g., ['Ordered boarding environment', 'According to "Beijing Rail Transit Ride']
    """
    mini_lines = []
    start_idx = 0
    
    for length in line_lengths:
        if start_idx >= len(text):
            # If text is fully split, remaining mini-lines are empty strings
            mini_lines.append('')
            continue
        
        end_idx = start_idx + length
        mini_line = text[start_idx:end_idx]
        mini_lines.append(mini_line)
        start_idx = end_idx
    
    return mini_lines


def test_mini_line_level(json_files, fusion_model_path, lm_model_path, vocab_path, 
                   map_file, ocr_char_file, mask_ratio=0.2, device=None, verbose=False):
    """
    Main function for mini-line level testing
    
    Workflow:
    1. Perform masking, LM generation, and fusion prediction at major line level
    2. Split major line into mini-lines according to segmentation information in JSON
    3. Calculate ACC, AR, CR metrics at mini-line level
    
    Args:
        json_files: List of JSON file paths
        fusion_model_path: Fusion model weight path
        lm_model_path: Bidirectional language model weight path
        vocab_path: LM vocabulary path (char_to_idx.json)
        map_file: OCR to LM mapping file
        ocr_char_file: OCR character list file
        mask_ratio: Mask ratio (default: 0.2)
        device: Device (auto-detected by default)
        verbose: Whether to print detailed information (default: False)
    
    Returns:
        results: Test results dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print("🧪 Mini-line Level Fusion Model Testing")
    print(f"{'='*80}")
    print(f"Device used: {device}")
    print(f"Mask ratio: {mask_ratio*100:.1f}%")
    print(f"{'='*80}\n")
    
    # Load models and mappings
    print("📦 Loading models and mappings...")
    
    # Load OCR character list and mappings
    ocr_to_lm, ocr_chars = load_mappings(map_file, ocr_char_file)
    print(f"  ✅ OCR character list: {len(ocr_chars)} characters")
    print(f"  ✅ OCR to LM mappings: {len(ocr_to_lm)} mappings")
    
    # Load bidirectional language model
    bidirectional_lm = BidirectionalProbabilityGenerator(
        model_path=lm_model_path,
        vocab_path=vocab_path,
        device=device
    )
    print(f"  ✅ Bidirectional language model loaded")
    
    # Load fusion model
    fusion_model = CrossAttentionFusion(
        feature_dim=FEATURE_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS
    ).to(device)
    
    state_dict = torch.load(fusion_model_path, map_location=device)
    fusion_model.load_state_dict(state_dict)
    fusion_model.eval()
    print(f"  ✅ Fusion model loaded")
    print()
    
    # Statistical data (mini-line level)
    stats = {
        'total_mini_lines': 0,  # Total number of mini-lines
        'correct_mini_lines': 0,  # Number of fully correct mini-lines (ACC) - Fusion model
        'total_chars': 0,     # Total number of characters
        'total_substitutions': 0,  # Total substitution errors - Fusion model
        'total_deletions': 0,      # Total deletion errors - Fusion model
        'total_insertions': 0,     # Total insertion errors - Fusion model
        # Original OCR statistics (mini-line level)
        'ocr_correct_mini_lines': 0,  # Number of fully correct mini-lines (ACC) - Original OCR
        'ocr_total_substitutions': 0,  # Total substitution errors - Original OCR
        'ocr_total_deletions': 0,      # Total deletion errors - Original OCR
        'ocr_total_insertions': 0,      # Total insertion errors - Original OCR
        'mini_line_details': [],         # Detailed information for each mini-line
        'warnings': {               # Warning statistics
            'mask_out_of_range': 0,      # Mask position out of range
            'char_replace_failed': 0,    # Character replacement failed
            'length_mismatch': 0,        # Length mismatch
            'missing_split_info': 0      # Missing segmentation information
        }
    }
    
    # Process all JSON files
    print(f"📂 Processing {len(json_files)} JSON files...\n")
    
    for json_file in json_files:
        print(f"Processing file: {os.path.basename(json_file)}")
        data = load_json_data(json_file)
        
        topk_probs_list = data['topk_probs']      # [batch_size, seq_len, 100]
        topk_indices_list = data['topk_indices'] # [batch_size, seq_len, 100]
        labels_list = data['labels']             # List of true labels
        decoded_texts_list = data['decoded_texts']  # List of decoded texts
        
        # Read mini-line segmentation information (if exists)
        preds_per_line_len = data.get('preds_per_line_len', None)  # Number of predicted characters per mini-line
        labels_per_line_len = data.get('labels_per_line_len', None)  # Number of label characters per mini-line
        lines_num = data.get('lines_num', None)  # Number of mini-lines
        
        batch_size = len(labels_list)
        
        for sample_idx in tqdm(range(batch_size), desc="  Processing samples"):
            decoded_text = decoded_texts_list[sample_idx]
            # Ensure decoded_text is string type
            if not isinstance(decoded_text, str):
                decoded_text = str(decoded_text)
            
            # Labels format: [["的", 1.0], ["惕", 1.0], ...], need to extract text string
            label_item = labels_list[sample_idx]
            if isinstance(label_item, list) and len(label_item) >= 1:
                true_text = label_item[0]  # Extract text string
            else:
                true_text = label_item  # Directly use if already string
            
            # Ensure true_text is string type
            if not isinstance(true_text, str):
                true_text = str(true_text)
            
            topk_probs = topk_probs_list[sample_idx]      # [seq_len, 100]
            topk_indices = topk_indices_list[sample_idx] # [seq_len, 100]
            
            # Skip empty text (only skip if true label is empty; empty prediction still participates in evaluation)
            if not true_text:
                continue
            
            # Ensure length consistency
            seq_len = len(decoded_text)
            true_len = len(true_text)

            # Special case: Empty predicted text but non-empty label, evaluate as empty string (independent of topk length)
            if seq_len == 0 and true_len > 0:
                mask_positions = []
                pred_text = ''

                # Check for mini-line segmentation information
                if preds_per_line_len is None or labels_per_line_len is None or \
                   len(preds_per_line_len) == 0 or len(labels_per_line_len) == 0:
                    # If no segmentation information, process as major line
                    # First perform normalization
                    pred_text_normalized = _unify_evaluation_text(pred_text)
                    true_text_normalized = _unify_evaluation_text(true_text)
                    ocr_text_normalized = _unify_evaluation_text(decoded_text)
                    
                    stats['total_mini_lines'] += 1
                    stats['total_chars'] += len(true_text_normalized)
                    
                    # Calculate original OCR metrics
                    ocr_is_correct = (ocr_text_normalized == true_text_normalized)
                    if ocr_is_correct:
                        stats['ocr_correct_mini_lines'] += 1
                    ocr_distance, ocr_substitutions, ocr_deletions, ocr_insertions = calculate_edit_distance(
                        ocr_text_normalized, true_text_normalized
                    )
                    stats['ocr_total_substitutions'] += ocr_substitutions
                    stats['ocr_total_deletions'] += ocr_deletions
                    stats['ocr_total_insertions'] += ocr_insertions
                    
                    # Calculate fusion model metrics
                    is_correct = (pred_text_normalized == true_text_normalized)
                    if is_correct:
                        stats['correct_mini_lines'] += 1
                    distance, substitutions, deletions, insertions = calculate_edit_distance(
                        pred_text_normalized, true_text_normalized
                    )
                    stats['total_substitutions'] += substitutions
                    stats['total_deletions'] += deletions
                    stats['total_insertions'] += insertions
                    
                    # Save detailed information
                    stats['mini_line_details'].append({
                        'sample_idx': sample_idx,
                        'mini_line_idx': 0,
                        'decoded_text': decoded_text,
                        'true_text': true_text,
                        'pred_text': pred_text,
                        'decoded_mini': ocr_text_normalized,
                        'true_mini': true_text_normalized,
                        'pred_mini': pred_text_normalized,
                        'is_correct': is_correct,
                        'ocr_is_correct': ocr_is_correct,
                        'mask_positions': mask_positions,
                        'num_masked': len(mask_positions),
                        'substitutions': substitutions,
                        'deletions': deletions,
                        'insertions': insertions,
                        'edit_distance': distance,
                        'ocr_substitutions': ocr_substitutions,
                        'ocr_deletions': ocr_deletions,
                        'ocr_insertions': ocr_insertions,
                        'ocr_edit_distance': ocr_distance,
                        'is_major_line': True
                    })
                else:
                    # With segmentation information, process as mini-lines
                    # First check segmentation information format
                    if (preds_per_line_len and len(preds_per_line_len) > 0 and 
                        isinstance(preds_per_line_len[0], list)):
                        # Each sample has its own segmentation information (list of lists)
                        if sample_idx < len(preds_per_line_len) and sample_idx < len(labels_per_line_len):
                            sample_preds_per_line_len = preds_per_line_len[sample_idx]
                            sample_labels_per_line_len = labels_per_line_len[sample_idx]
                        else:
                            # Index out of range, skip this sample
                            continue
                    elif (preds_per_line_len and labels_per_line_len and 
                          len(preds_per_line_len) > 0 and len(labels_per_line_len) > 0):
                        # All samples share segmentation information (simple list)
                        sample_preds_per_line_len = preds_per_line_len
                        sample_labels_per_line_len = labels_per_line_len
                    else:
                        # No valid segmentation information, skip
                        continue
                    
                    # First split major line into mini-lines (using raw text, unnormalized)
                    pred_mini_lines_raw = split_into_mini_lines(pred_text, sample_preds_per_line_len)
                    true_mini_lines_raw = split_into_mini_lines(true_text, sample_labels_per_line_len)
                    ocr_mini_lines_raw = split_into_mini_lines(decoded_text, sample_preds_per_line_len)
                    
                    num_mini_lines = len(sample_labels_per_line_len)
                    for mini_line_idx in range(num_mini_lines):
                        # Get raw mini-line text
                        pred_mini_raw = pred_mini_lines_raw[mini_line_idx] if mini_line_idx < len(pred_mini_lines_raw) else ''
                        true_mini_raw = true_mini_lines_raw[mini_line_idx] if mini_line_idx < len(true_mini_lines_raw) else ''
                        ocr_mini_raw = ocr_mini_lines_raw[mini_line_idx] if mini_line_idx < len(ocr_mini_lines_raw) else ''
                        
                        # Normalize each mini-line separately
                        pred_mini = _unify_evaluation_text(pred_mini_raw)
                        true_mini = _unify_evaluation_text(true_mini_raw)
                        ocr_mini = _unify_evaluation_text(ocr_mini_raw)
                        
                        stats['total_mini_lines'] += 1
                        stats['total_chars'] += len(true_mini)
                        
                        # Calculate metrics
                        ocr_is_correct = (ocr_mini == true_mini)
                        if ocr_is_correct:
                            stats['ocr_correct_mini_lines'] += 1
                        ocr_distance, ocr_substitutions, ocr_deletions, ocr_insertions = calculate_edit_distance(
                            ocr_mini, true_mini
                        )
                        stats['ocr_total_substitutions'] += ocr_substitutions
                        stats['ocr_total_deletions'] += ocr_deletions
                        stats['ocr_total_insertions'] += ocr_insertions
                        
                        is_correct = (pred_mini == true_mini)
                        if is_correct:
                            stats['correct_mini_lines'] += 1
                        distance, substitutions, deletions, insertions = calculate_edit_distance(
                            pred_mini, true_mini
                        )
                        stats['total_substitutions'] += substitutions
                        stats['total_deletions'] += deletions
                        stats['total_insertions'] += insertions
                        
                        stats['mini_line_details'].append({
                            'sample_idx': sample_idx,
                            'mini_line_idx': mini_line_idx,
                            'decoded_text': decoded_text,
                            'true_text': true_text,
                            'pred_text': pred_text,
                            'decoded_mini': ocr_mini,
                            'true_mini': true_mini,
                            'pred_mini': pred_mini,
                            'is_correct': is_correct,
                            'ocr_is_correct': ocr_is_correct,
                            'mask_positions': mask_positions,
                            'num_masked': len(mask_positions),
                            'substitutions': substitutions,
                            'deletions': deletions,
                            'insertions': insertions,
                            'edit_distance': distance,
                            'ocr_substitutions': ocr_substitutions,
                            'ocr_deletions': ocr_deletions,
                            'ocr_insertions': ocr_insertions,
                            'ocr_edit_distance': ocr_distance
                        })

                # Proceed to next sample
                continue
            
            # Check lengths of topk_probs and topk_indices
            if seq_len != len(topk_probs) or seq_len != len(topk_indices):
                # Length mismatch, skip this sample
                continue
            
            # Note: Lengths of decoded_text and true_text may be inconsistent
            # This is normal because OCR recognition errors can cause length differences
            # We will handle this when calculating edit distance
            
            # 1. Find mask positions
            mask_positions = find_mask_positions(topk_probs, mask_ratio)
            
            if not mask_positions:
                # If no mask positions, use original text directly
                pred_text = decoded_text
            else:
                # 2. Generate LM probabilities
                lm_c100_dict = generate_lm_probs_for_masks(
                    decoded_text, mask_positions, bidirectional_lm, 
                    ocr_to_lm, topk_indices
                )
                
                # 3. Fusion prediction and replacement
                pred_text_list, predictions = fuse_and_replace(
                    decoded_text, mask_positions, topk_probs, topk_indices,
                    lm_c100_dict, fusion_model, ocr_chars, device
                )
                pred_text = ''.join(pred_text_list)
            
            # 4. Check for mini-line segmentation information
            # Note: preds_per_line_len and labels_per_line_len are for the entire batch, need to get by sample_idx
            # If these fields do not exist in JSON, or format is incorrect, or empty list, process as major line
            if preds_per_line_len is None or labels_per_line_len is None or \
               len(preds_per_line_len) == 0 or len(labels_per_line_len) == 0:
                # If no segmentation information, process as major line (compatible with old format)
                # First perform normalization
                pred_text_normalized = _unify_evaluation_text(pred_text)
                true_text_normalized = _unify_evaluation_text(true_text)
                ocr_text_normalized = _unify_evaluation_text(decoded_text)
                
                stats['warnings']['missing_split_info'] += 1
                # Use length of normalized true text as total number of characters
                stats['total_chars'] += len(true_text_normalized)
                stats['total_mini_lines'] += 1
                
                # Calculate original OCR metrics
                ocr_is_correct = (ocr_text_normalized == true_text_normalized)
                if ocr_is_correct:
                    stats['ocr_correct_mini_lines'] += 1
                ocr_distance, ocr_substitutions, ocr_deletions, ocr_insertions = calculate_edit_distance(
                    ocr_text_normalized, true_text_normalized
                )
                stats['ocr_total_substitutions'] += ocr_substitutions
                stats['ocr_total_deletions'] += ocr_deletions
                stats['ocr_total_insertions'] += ocr_insertions
                
                # Calculate fusion model metrics
                is_correct = (pred_text_normalized == true_text_normalized)
                if is_correct:
                    stats['correct_mini_lines'] += 1
                distance, substitutions, deletions, insertions = calculate_edit_distance(
                    pred_text_normalized, true_text_normalized
                )
                stats['total_substitutions'] += substitutions
                stats['total_deletions'] += deletions
                stats['total_insertions'] += insertions
                
                # Save detailed information
                stats['mini_line_details'].append({
                    'sample_idx': sample_idx,
                    'mini_line_idx': 0,
                    'decoded_text': decoded_text,
                    'true_text': true_text,
                    'pred_text': pred_text,
                    'true_text_normalized': true_text_normalized,
                    'pred_text_normalized': pred_text_normalized,
                    'ocr_text_normalized': ocr_text_normalized,
                    'is_correct': is_correct,
                    'ocr_is_correct': ocr_is_correct,
                    'mask_positions': mask_positions,
                    'num_masked': len(mask_positions),
                    'substitutions': substitutions,
                    'deletions': deletions,
                    'insertions': insertions,
                    'edit_distance': distance,
                    'ocr_substitutions': ocr_substitutions,
                    'ocr_deletions': ocr_deletions,
                    'ocr_insertions': ocr_insertions,
                    'ocr_edit_distance': ocr_distance,
                    'is_major_line': True  # Mark as major line (no segmentation)
                })
                continue
            
            # 5. If there is segmentation information, first split into mini-lines (using raw text, unnormalized)
            # Note: preds_per_line_len and labels_per_line_len may be list of lists (one list per sample)
            # Or a single list (shared by all samples, which is the most common case)
            # First check if it's list of lists (one list per sample)
            if (preds_per_line_len and len(preds_per_line_len) > 0 and 
                isinstance(preds_per_line_len[0], list)):
                # Each sample has its own segmentation information (list of lists)
                if sample_idx < len(preds_per_line_len) and sample_idx < len(labels_per_line_len):
                    sample_preds_per_line_len = preds_per_line_len[sample_idx]
                    sample_labels_per_line_len = labels_per_line_len[sample_idx]
                else:
                    # Index out of range, skip this sample
                    stats['warnings']['length_mismatch'] += 1
                    if verbose:
                        print(f"  ⚠️ Sample {sample_idx}: Segmentation information index out of range")
                    continue
            elif (preds_per_line_len and labels_per_line_len and 
                  len(preds_per_line_len) > 0 and len(labels_per_line_len) > 0):
                # All samples share segmentation information (simple list, most common case)
                # Note: In this case, all samples should use the same segmentation information
                sample_preds_per_line_len = preds_per_line_len
                sample_labels_per_line_len = labels_per_line_len
            else:
                # No segmentation information, should have been handled earlier, should not reach here
                stats['warnings']['missing_split_info'] += 1
                if verbose:
                    print(f"  ⚠️ Sample {sample_idx}: Missing segmentation information but code logic error")
                continue
            
            # First split major line into mini-lines (using raw text, unnormalized)
            pred_mini_lines_raw = split_into_mini_lines(pred_text, sample_preds_per_line_len)
            true_mini_lines_raw = split_into_mini_lines(true_text, sample_labels_per_line_len)
            ocr_mini_lines_raw = split_into_mini_lines(decoded_text, sample_preds_per_line_len)
            
            # Ensure consistent number of mini-lines
            num_mini_lines = len(sample_preds_per_line_len)
            if len(pred_mini_lines_raw) != num_mini_lines or len(true_mini_lines_raw) != num_mini_lines:
                stats['warnings']['length_mismatch'] += 1
                if verbose:
                    print(f"  ⚠️ Sample {sample_idx}: Inconsistent number of mini-lines (pred:{len(pred_mini_lines_raw)}, true:{len(true_mini_lines_raw)}, expected:{num_mini_lines})")
                continue
            
            # 6. Normalize each mini-line separately
            # 7. Calculate metrics for each mini-line
            for mini_line_idx in range(num_mini_lines):
                # Get raw mini-line text
                pred_mini_raw = pred_mini_lines_raw[mini_line_idx]
                true_mini_raw = true_mini_lines_raw[mini_line_idx]
                ocr_mini_raw = ocr_mini_lines_raw[mini_line_idx] if mini_line_idx < len(ocr_mini_lines_raw) else ''
                
                # Normalize each mini-line separately
                pred_mini = _unify_evaluation_text(pred_mini_raw)
                true_mini = _unify_evaluation_text(true_mini_raw)
                ocr_mini = _unify_evaluation_text(ocr_mini_raw)
                
                stats['total_mini_lines'] += 1
                stats['total_chars'] += len(true_mini)
                
                # Calculate original OCR metrics
                ocr_is_correct = (ocr_mini == true_mini)
                if ocr_is_correct:
                    stats['ocr_correct_mini_lines'] += 1
                ocr_distance, ocr_substitutions, ocr_deletions, ocr_insertions = calculate_edit_distance(
                    ocr_mini, true_mini
                )
                stats['ocr_total_substitutions'] += ocr_substitutions
                stats['ocr_total_deletions'] += ocr_deletions
                stats['ocr_total_insertions'] += ocr_insertions
                
                # Calculate fusion model metrics
                is_correct = (pred_mini == true_mini)
                if is_correct:
                    stats['correct_mini_lines'] += 1
                distance, substitutions, deletions, insertions = calculate_edit_distance(
                    pred_mini, true_mini
                )
                stats['total_substitutions'] += substitutions
                stats['total_deletions'] += deletions
                stats['total_insertions'] += insertions
                
                # Save mini-line detailed information
                stats['mini_line_details'].append({
                    'sample_idx': sample_idx,
                    'mini_line_idx': mini_line_idx,
                    'decoded_text': decoded_text,
                    'true_text': true_text,
                    'pred_text': pred_text,
                    'decoded_mini': ocr_mini,
                    'true_mini': true_mini,
                    'pred_mini': pred_mini,
                    'is_correct': is_correct,
                    'ocr_is_correct': ocr_is_correct,
                    'mask_positions': mask_positions,
                    'num_masked': len(mask_positions),
                    'substitutions': substitutions,
                    'deletions': deletions,
                    'insertions': insertions,
                    'edit_distance': distance,
                    'ocr_substitutions': ocr_substitutions,
                    'ocr_deletions': ocr_deletions,
                    'ocr_insertions': ocr_insertions,
                    'ocr_edit_distance': ocr_distance
                })
                
                # Print mini-line detailed information (for debugging)
                if verbose:
                    status = "✅" if is_correct else "❌"
                    ocr_status = "✅" if ocr_is_correct else "❌"
                    print(f"\n{'─'*80}")
                    print(f"{status} Sample #{sample_idx} Mini-line #{mini_line_idx} (Major line masked: {len(mask_positions)} positions)")
                    print(f"{'─'*80}")
                    print(f"📝 Major line raw text:")
                    print(f"  OCR raw: {decoded_text}")
                    print(f"  Fusion pred: {pred_text}")
                    print(f"  True: {true_text}")
                    print(f"📝 Mini-line text:")
                    print(f"  OCR raw: {ocr_mini}")
                    print(f"  Fusion pred: {pred_mini}")
                    print(f"  True: {true_mini}")
                    print(f"📊 Original OCR metrics: {ocr_status}")
                    print(f"  Edit distance: {ocr_distance} | Substitutions:{ocr_substitutions} | Deletions:{ocr_deletions} | Insertions:{ocr_insertions}")
                    print(f"📊 Fusion model metrics: {status}")
                    print(f"  Edit distance: {distance} | Substitutions:{substitutions} | Deletions:{deletions} | Insertions:{insertions}")
                    if not is_correct and distance > 0:
                        pred_len = len(pred_mini)
                        true_len = len(true_mini)
                        if pred_len != true_len:
                            print(f"  Length difference: Predicted={pred_len}, True={true_len}")
                    print()
    
    # Calculate final metrics (mini-line level)
    total_mini_lines = stats['total_mini_lines']
    total_chars = stats['total_chars']
    
    # Fusion model metrics
    # ACC: Mini-line level accuracy
    acc = stats['correct_mini_lines'] / total_mini_lines if total_mini_lines > 0 else 0.0
    
    # AR/CR: Standard definition
    # AR = (N_t - D_e - S_e - I_e) / N_t
    # CR = (N_t - D_e - S_e) / N_t
    N_t = total_chars
    D_e = stats['total_deletions']
    S_e = stats['total_substitutions']
    I_e = stats['total_insertions']
    
    ar = (N_t - D_e - S_e - I_e) / N_t if N_t > 0 else 0.0
    cr = (N_t - D_e - S_e) / N_t if N_t > 0 else 0.0
    
    # Original OCR metrics
    ocr_acc = stats['ocr_correct_mini_lines'] / total_mini_lines if total_mini_lines > 0 else 0.0
    ocr_D_e = stats['ocr_total_deletions']
    ocr_S_e = stats['ocr_total_substitutions']
    ocr_I_e = stats['ocr_total_insertions']
    
    ocr_ar = (N_t - ocr_D_e - ocr_S_e - ocr_I_e) / N_t if N_t > 0 else 0.0
    ocr_cr = (N_t - ocr_D_e - ocr_S_e) / N_t if N_t > 0 else 0.0
    
    results = {
        'total_mini_lines': total_mini_lines,
        'total_chars': total_chars,
        # Fusion model metrics
        'correct_mini_lines': stats['correct_mini_lines'],
        'acc': acc,
        'ar': ar,
        'cr': cr,
        'total_substitutions': S_e,
        'total_deletions': D_e,
        'total_insertions': I_e,
        # Original OCR metrics
        'ocr_correct_mini_lines': stats['ocr_correct_mini_lines'],
        'ocr_acc': ocr_acc,
        'ocr_ar': ocr_ar,
        'ocr_cr': ocr_cr,
        'ocr_total_substitutions': ocr_S_e,
        'ocr_total_deletions': ocr_D_e,
        'ocr_total_insertions': ocr_I_e,
        'stats': stats
    }
    
    return results


def print_results(results):
    """Print test results"""
    print(f"\n{'='*80}")
    print("📊 Test Results (Mini-line Level)")
    print(f"{'='*80}\n")
    
    print(f"Total tested mini-lines: {results['total_mini_lines']:,}")
    print(f"Total characters: {results['total_chars']:,}\n")
    
    print(f"{'='*80}")
    print("🎯 Core Metrics Comparison")
    print(f"{'='*80}")
    print(f"{'Metric':<12} {'Original OCR':<20} {'Fusion Model':<20} {'Improvement':<15}")
    print(f"{'-'*80}")
    
    # ACC comparison
    acc_improvement = results['acc'] - results['ocr_acc']
    acc_improvement_pct = acc_improvement * 100
    acc_sign = "↑" if acc_improvement > 0 else "↓" if acc_improvement < 0 else "="
    print(f"{'ACC':<12} {results['ocr_acc']:.4f} ({results['ocr_acc']*100:.2f}%){'':<6} "
          f"{results['acc']:.4f} ({results['acc']*100:.2f}%){'':<6} "
          f"{acc_sign} {abs(acc_improvement_pct):.2f}%")
    
    # AR comparison
    ar_improvement = results['ar'] - results['ocr_ar']
    ar_improvement_pct = ar_improvement * 100
    ar_sign = "↑" if ar_improvement > 0 else "↓" if ar_improvement < 0 else "="
    print(f"{'AR':<12} {results['ocr_ar']:.4f} ({results['ocr_ar']*100:.2f}%){'':<6} "
          f"{results['ar']:.4f} ({results['ar']*100:.2f}%){'':<6} "
          f"{ar_sign} {abs(ar_improvement_pct):.2f}%")
    
    # CR comparison
    cr_improvement = results['cr'] - results['ocr_cr']
    cr_improvement_pct = cr_improvement * 100
    cr_sign = "↑" if cr_improvement > 0 else "↓" if cr_improvement < 0 else "="
    print(f"{'CR':<12} {results['ocr_cr']:.4f} ({results['ocr_cr']*100:.2f}%){'':<6} "
          f"{results['cr']:.4f} ({results['cr']*100:.2f}%){'':<6} "
          f"{cr_sign} {abs(cr_improvement_pct):.2f}%")
    
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print("📐 Fusion Model AR/CR Calculation Details")
    print(f"{'='*80}")
    print(f"  N_t (Total characters):        {results['total_chars']:,}")
    print(f"  D_e (Deletion errors):      {results['total_deletions']:,}")
    print(f"  S_e (Substitution errors):      {results['total_substitutions']:,}")
    print(f"  I_e (Insertion errors):      {results['total_insertions']:,}")
    print(f"  ")
    print(f"  AR = (N_t - D_e - S_e - I_e) / N_t")
    print(f"     = ({results['total_chars']:,} - {results['total_deletions']:,} - {results['total_substitutions']:,} - {results['total_insertions']:,}) / {results['total_chars']:,}")
    print(f"     = {results['ar']:.4f}")
    print(f"  ")
    print(f"  CR = (N_t - D_e - S_e) / N_t")
    print(f"     = ({results['total_chars']:,} - {results['total_deletions']:,} - {results['total_substitutions']:,}) / {results['total_chars']:,}")
    print(f"     = {results['cr']:.4f}")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print("📐 Original OCR AR/CR Calculation Details")
    print(f"{'='*80}")
    print(f"  N_t (Total characters):        {results['total_chars']:,}")
    print(f"  D_e (Deletion errors):      {results['ocr_total_deletions']:,}")
    print(f"  S_e (Substitution errors):      {results['ocr_total_substitutions']:,}")
    print(f"  I_e (Insertion errors):      {results['ocr_total_insertions']:,}")
    print(f"  ")
    print(f"  AR = (N_t - D_e - S_e - I_e) / N_t")
    print(f"     = ({results['total_chars']:,} - {results['ocr_total_deletions']:,} - {results['ocr_total_substitutions']:,} - {results['ocr_total_insertions']:,}) / {results['total_chars']:,}")
    print(f"     = {results['ocr_ar']:.4f}")
    print(f"  ")
    print(f"  CR = (N_t - D_e - S_e) / N_t")
    print(f"     = ({results['total_chars']:,} - {results['ocr_total_deletions']:,} - {results['ocr_total_substitutions']:,}) / {results['total_chars']:,}")
    print(f"     = {results['ocr_cr']:.4f}")
    print(f"{'='*80}\n")


def save_results(results, output_path):
    """Save test results to JSON file"""
    # Prepare data to save (remove detailed information to reduce file size)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'total_mini_lines': int(results['total_mini_lines']),
        'total_chars': int(results['total_chars']),
        # Fusion model metrics
        'fusion_model': {
            'correct_mini_lines': int(results['correct_mini_lines']),
            'metrics': {
                'acc': float(results['acc']),
                'ar': float(results['ar']),
                'cr': float(results['cr']),
            },
            'errors': {
                'substitutions': int(results['total_substitutions']),
                'deletions': int(results['total_deletions']),
                'insertions': int(results['total_insertions']),
            },
            'calculation': {
                'N_t': int(results['total_chars']),
                'D_e': int(results['total_deletions']),
                'S_e': int(results['total_substitutions']),
                'I_e': int(results['total_insertions']),
            }
        },
        # Original OCR metrics
        'original_ocr': {
            'correct_mini_lines': int(results['ocr_correct_mini_lines']),
            'metrics': {
                'acc': float(results['ocr_acc']),
                'ar': float(results['ocr_ar']),
                'cr': float(results['ocr_cr']),
            },
            'errors': {
                'substitutions': int(results['ocr_total_substitutions']),
                'deletions': int(results['ocr_total_deletions']),
                'insertions': int(results['ocr_total_insertions']),
            },
            'calculation': {
                'N_t': int(results['total_chars']),
                'D_e': int(results['ocr_total_deletions']),
                'S_e': int(results['ocr_total_substitutions']),
                'I_e': int(results['ocr_total_insertions']),
            }
        },
        # Improvement comparison
        'improvement': {
            'acc': float(results['acc'] - results['ocr_acc']),
            'ar': float(results['ar'] - results['ocr_ar']),
            'cr': float(results['cr'] - results['ocr_cr']),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Test results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Mini-line Level Fusion Model Testing Script')
    parser.add_argument('--json_dir', type=str, required=True,
                       help='JSON file directory')
    parser.add_argument('--json_pattern', type=str, default='*.json',
                       help='JSON file matching pattern (default: *.json)')
    parser.add_argument('--fusion_model', type=str, 
                       default='./models_svtr_image_3/fusion_model_best.pth',
                       help='Fusion model weight path')
    parser.add_argument('--lm_model', type=str,
                       default='./language_model/checkpoint_epoch_14.pth',
                       help='Bidirectional language model weight path')
    parser.add_argument('--vocab_path', type=str,
                       default='./data/char_to_idx.json',
                       help='LM vocabulary path')
    parser.add_argument('--map_file', type=str,
                       default='./map.json',
                       help='OCR to LM mapping file')
    parser.add_argument('--ocr_char_file', type=str,
                       default='./ppocr_keys_v1.txt',
                       help='OCR character list file')
    parser.add_argument('--mask_ratio', type=float, default=0.2,
                       help='Mask ratio (default: 0.2, i.e., lowest 20%)')
    parser.add_argument('--output_dir', type=str, default='./output_1',
                       help='Test results save directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Print normalized predicted text for each mini-line (for debugging)')
     
    args = parser.parse_args()
    
    # Scan JSON files
    pattern = os.path.join(args.json_dir, args.json_pattern)
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"❌ No JSON files found: {pattern}")
        return
    
    print(f"🔍 Found {len(json_files)} JSON files")
    
    # Run test
    results = test_mini_line_level(
        json_files=json_files,
        fusion_model_path=args.fusion_model,
        lm_model_path=args.lm_model,
        vocab_path=args.vocab_path,
        map_file=args.map_file,
        ocr_char_file=args.ocr_char_file,
        mask_ratio=args.mask_ratio,
        verbose=args.verbose
    )
    
    # Print results
    print_results(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(args.output_dir, f"mini_line_level_test_{timestamp}.json")
    save_results(results, output_json)
    
    print(f"\n{'='*80}")
    print("🎉 Testing completed!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()