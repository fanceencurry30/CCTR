#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Normalization Preprocessing Script
Purpose: Normalize original OCR and LM probability sequences to sum to 1.0
Input: Original *_formatted.pt files
Output: Normalized *_normalized.pt files
"""

import torch
import os
from tqdm import tqdm
import glob


def normalize_pt_file(input_path, output_path):
    """
    Normalize a single .pt file
    
    Args:
        input_path: Path to original .pt file
        output_path: Path to output .pt file
    """
    print(f"\nProcessing file: {os.path.basename(input_path)}")
    
    # Load data
    data = torch.load(input_path)
    
    ocr_c100 = data['ocr_c100']  # [N, 1, 100]
    lm_c100 = data['lm_c100']    # [N, 1, 100]
    gt_c100 = data['gt_c100']    # [N, 1, 100]
    
    # Calculate sum of original probabilities
    ocr_sum_before = ocr_c100.sum(dim=-1).mean().item()
    lm_sum_before = lm_c100.sum(dim=-1).mean().item()
    
    print(f"  Original OCR probability sum: {ocr_sum_before:.6f}")
    print(f"  Original LM probability sum:  {lm_sum_before:.6f}")
    
    # Normalize OCR probabilities
    ocr_c100_normalized = ocr_c100 / (ocr_c100.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Normalize LM probabilities
    lm_c100_normalized = lm_c100 / (lm_c100.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Verify normalization effect
    ocr_sum_after = ocr_c100_normalized.sum(dim=-1).mean().item()
    lm_sum_after = lm_c100_normalized.sum(dim=-1).mean().item()
    
    print(f"  Normalized OCR probability sum: {ocr_sum_after:.6f}")
    print(f"  Normalized LM probability sum:  {lm_sum_after:.6f}")
    
    # Check for abnormal values
    if torch.isnan(ocr_c100_normalized).any() or torch.isnan(lm_c100_normalized).any():
        print(f"  ⚠️ Warning: NaN values exist after normalization!")
        return False
    
    if torch.isinf(ocr_c100_normalized).any() or torch.isinf(lm_c100_normalized).any():
        print(f"  ⚠️ Warning: Inf values exist after normalization!")
        return False
    
    # Save normalized data (retain all fields from original data)
    normalized_data = {
        'ocr_c100': ocr_c100_normalized,
        'lm_c100': lm_c100_normalized,
        'gt_c100': gt_c100  # GT does not need normalization
    }
    
    # Retain other fields from original data (e.g., topk_indices, stats, etc.)
    if 'topk_indices' in data:
        normalized_data['topk_indices'] = data['topk_indices']
        print(f"  ├─ Retained topk_indices: {data['topk_indices'].shape}")
    
    if 'stats' in data:
        normalized_data['stats'] = data['stats']
        print(f"  ├─ Retained stats information")
    
    # Retain other possible fields
    for key in data.keys():
        if key not in ['ocr_c100', 'lm_c100', 'gt_c100', 'topk_indices', 'stats']:
            normalized_data[key] = data[key]
            print(f"  ├─ Retained additional field: {key}")
    
    torch.save(normalized_data, output_path)
    
    # Check file sizes
    original_size = os.path.getsize(input_path) / (1024**3)
    normalized_size = os.path.getsize(output_path) / (1024**3)
    
    print(f"  Original file size: {original_size:.2f} GB")
    print(f"  Normalized file size: {normalized_size:.2f} GB")
    print(f"  ✅ Normalization completed!")
    
    return True


def batch_normalize(input_dir, output_dir, pattern="*_formatted.pt"):
    """
    Batch normalize all .pt files in the directory
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        pattern: File matching pattern
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    input_pattern = os.path.join(input_dir, pattern)
    pt_files = glob.glob(input_pattern)
    
    if not pt_files:
        print(f"❌ Error: No files matching {pattern} found in {input_dir}!")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(pt_files)} files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    success_count = 0
    fail_count = 0
    
    for pt_file in pt_files:
        # Generate output file name
        basename = os.path.basename(pt_file)
        # Replace _formatted.pt with _normalized.pt
        output_basename = basename.replace('_formatted.pt', '_normalized.pt')
        output_path = os.path.join(output_dir, output_basename)
        
        try:
            success = normalize_pt_file(pt_file, output_path)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fail_count += 1
    
    print(f"\n{'='*80}")
    print(f"Processing completed!")
    print(f"  Success: {success_count} files")
    print(f"  Failed: {fail_count} files")
    print(f"{'='*80}\n")


def verify_normalized_data(pt_file_path):
    """
    Verify the quality of normalized data
    
    Args:
        pt_file_path: Path to .pt file
    """
    print(f"\nVerifying file: {os.path.basename(pt_file_path)}")
    
    data = torch.load(pt_file_path)
    
    ocr_c100 = data['ocr_c100']
    lm_c100 = data['lm_c100']
    gt_c100 = data['gt_c100']
    
    # Statistical information
    print(f"\nData shapes:")
    print(f"  ocr_c100: {ocr_c100.shape}")
    print(f"  lm_c100:  {lm_c100.shape}")
    print(f"  gt_c100:  {gt_c100.shape}")
    
    # Check additional fields
    if 'topk_indices' in data:
        print(f"  topk_indices: {data['topk_indices'].shape} ✅")
    else:
        print(f"  topk_indices: ❌ Missing (Warning: Cannot restore real characters)")
    
    if 'stats' in data:
        print(f"  stats: ✅ Exists")
    else:
        print(f"  stats: ⚠️ Does not exist")
    
    # Display other additional fields
    extra_keys = [k for k in data.keys() if k not in ['ocr_c100', 'lm_c100', 'gt_c100', 'topk_indices', 'stats']]
    if extra_keys:
        print(f"  Other fields: {', '.join(extra_keys)}")
    
    # Probability sum statistics
    ocr_sums = ocr_c100.sum(dim=-1)
    lm_sums = lm_c100.sum(dim=-1)
    
    print(f"\nOCR probability sum statistics:")
    print(f"  Mean: {ocr_sums.mean():.6f}")
    print(f"  Min:  {ocr_sums.min():.6f}")
    print(f"  Max:  {ocr_sums.max():.6f}")
    print(f"  Std:  {ocr_sums.std():.6f}")
    
    print(f"\nLM probability sum statistics:")
    print(f"  Mean: {lm_sums.mean():.6f}")
    print(f"  Min:  {lm_sums.min():.6f}")
    print(f"  Max:  {lm_sums.max():.6f}")
    print(f"  Std:  {lm_sums.std():.6f}")
    
    # Check for abnormal values
    has_nan = torch.isnan(ocr_c100).any() or torch.isnan(lm_c100).any()
    has_inf = torch.isinf(ocr_c100).any() or torch.isinf(lm_c100).any()
    
    print(f"\nData quality check:")
    print(f"  Contains NaN: {'❌ Yes' if has_nan else '✅ No'}")
    print(f"  Contains Inf: {'❌ Yes' if has_inf else '✅ No'}")
    
    # Check probability range
    ocr_in_range = (ocr_c100 >= 0).all() and (ocr_c100 <= 1).all()
    lm_in_range = (lm_c100 >= 0).all() and (lm_c100 <= 1).all()
    
    print(f"  OCR probabilities in [0,1]: {'✅ Yes' if ocr_in_range else '❌ No'}")
    print(f"  LM probabilities in [0,1]:  {'✅ Yes' if lm_in_range else '❌ No'}")
    
    # Sample display
    print(f"\nRandomly sample 3 samples:")
    for i in range(min(3, ocr_c100.size(0))):
        idx = torch.randint(0, ocr_c100.size(0), (1,)).item()
        print(f"\n  Sample {i+1} (index={idx}):")
        print(f"    OCR Top3: {ocr_c100[idx, 0].topk(3)}")
        print(f"    LM  Top3: {lm_c100[idx, 0].topk(3)}")
        print(f"    GT  Class: {gt_c100[idx, 0].argmax().item()}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Configure paths
    INPUT_DIR = "./new_pt_data/nrtr/train"
    OUTPUT_DIR = "./new_pt_data/nrtr/train_norm"
    
    # Batch normalization
    batch_normalize(INPUT_DIR, OUTPUT_DIR, pattern="*.pt")
    
    # Verify the first normalized file
    normalized_files = glob.glob(os.path.join(OUTPUT_DIR, "*_normalized.pt"))
    if normalized_files:
        verify_normalized_data(normalized_files[0])