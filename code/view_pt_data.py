"""
View Fusion Training Data (.pt File)

Core Concepts:
- OCR_100: Probabilities of the top-100 candidate characters predicted by the OCR model
- LM_100:  Corresponding probabilities of **the same 100 candidates** in the language model (one-to-one correspondence)
- GT_100:  One-hot encoding of the ground truth character (among the 100 candidates)
- topk_indices: Indices of OCR's top-100 candidates (defines which 100 characters they are)

Important Notes:
- LM_100 is not the language model's own top-100 predictions
- LM_100 are probabilities of OCR's 100 candidates queried from the language model
- The three probability vectors (OCR/LM/GT) all correspond to the same 100 candidate characters

Features:
1. Display basic file information and statistical data
2. Interactively view ocr_c100, lm_c100, gt_c100 of any character sample
3. Show top-k values of probability distributions
4. Verify data quality and one-to-one correspondence

Author: Modified based on view_json_data.py
Date: 2025
"""

import torch
import numpy as np
import json
import sys
import os


def load_pt_file(pt_file):
    """Load .pt file"""
    print(f"Loading: {pt_file}...")
    data = torch.load(pt_file, map_location='cpu')
    print("✅ Loaded successfully!\n")
    return data


def display_file_info(data):
    """Display basic file information"""
    print("="*80)
    print("📦 Basic File Information")
    print("="*80)
    
    # Check data structure
    if isinstance(data, dict):
        print(f"Data Type: Dictionary")
        print(f"Keys Included: {list(data.keys())}\n")
        
        # Display tensor shapes
        if 'ocr_c100' in data:
            print(f"OCR Probability Tensor Shape: {data['ocr_c100'].shape}")
        if 'lm_c100' in data:
            print(f"LM  Probability Tensor Shape: {data['lm_c100'].shape}")
        if 'gt_c100' in data:
            print(f"GT  Probability Tensor Shape: {data['gt_c100'].shape}")
        if 'topk_indices' in data:
            print(f"Top-100 Indices Shape: {data['topk_indices'].shape}")
        
        # Display total number of samples
        if 'ocr_c100' in data:
            total_samples = data['ocr_c100'].shape[0]
            print(f"\n✅ Total Character Samples: {total_samples:,}")
        
        # Display statistical information
        if 'stats' in data:
            print("\n" + "="*80)
            print("📊 Data Statistics")
            print("="*80)
            stats = data['stats']
            
            print("\n【Stage 1 Filtering: Initial Screening】")
            print(f"  ├─ Total Samples (Documents): {stats.get('stage1_total_samples', 0):,}")
            print(f"  ├─ Filtered Samples:")
            print(f"  │  ├─ Empty Decoding: {stats.get('stage1_filtered_empty', 0):,}")
            print(f"  │  ├─ Type Check Failed: {stats.get('stage1_filtered_type_error', 0):,}")
            print(f"  │  └─ Length Mismatch: {stats.get('stage1_filtered_length_mismatch', 0):,}")
            print(f"  └─ ✅ Retained Samples: {stats.get('stage1_retained', 0):,}")
            
            print("\n【Stage 2: Intelligent Masking Strategy】")
            print(f"  ├─ Total Characters: {stats.get('stage2_total_chars', 0):,}")
            print(f"  ├─ OCR Incorrectly Predicted Characters: {stats.get('stage2_incorrect_chars', 0):,}")
            print(f"  ├─ OCR Correctly Predicted Characters: {stats.get('stage2_correct_chars', 0):,}")
            print(f"  ├─ Total Masked Characters: {stats.get('stage2_masked_chars', 0):,}")
            print(f"  │  ├─ Masked & OCR Incorrect: {stats.get('stage2_masked_incorrect', 0):,}")
            print(f"  │  └─ Masked & OCR Correct: {stats.get('stage2_masked_correct', 0):,}")
            
            print("\n【Stage 3: GT Filtering & Final Sample Generation】")
            print(f"  ├─ Filtered (GT not in OCR top-100): {stats.get('stage3_filtered_gt_not_in_top100', 0):,}")
            print(f"  └─ ✅ Final Character-Level Samples: {stats.get('stage3_final_chars', 0):,}")
            final_chars = stats.get('stage3_final_chars', 1)  # Avoid division by zero
            print(f"     ├─ 🟢 OCR Correct: {stats.get('stage3_final_correct', 0):,} "
                  f"({stats.get('stage3_final_correct', 0)/final_chars*100:.2f}%)")
            print(f"     └─ 🔴 OCR Incorrect: {stats.get('stage3_final_incorrect', 0):,} "
                  f"({stats.get('stage3_final_incorrect', 0)/final_chars*100:.2f}%)")
    else:
        print(f"Data Type: {type(data)}")
    
    print("="*80 + "\n")


def load_ocr_chars(ocr_char_file):
    """Load OCR character list"""
    try:
        with open(ocr_char_file, 'r', encoding='utf-8') as f:
            ocr_chars = [line.strip() for line in f.readlines()]
        return ocr_chars
    except FileNotFoundError:
        print(f"⚠️ Warning: OCR character list file not found - {ocr_char_file}")
        return None


def verify_data_consistency(ocr_probs, lm_probs, gt_probs, topk_indices):
    """
    Verify data consistency
    
    Check Items:
    1. Whether the lengths of the three probability vectors are consistent
    2. Whether the length of topk_indices is consistent with probability vectors
    3. Whether the sum of GT probabilities is 1.0
    
    Returns:
        bool: Whether data is consistent
        str: Error message (if any)
    """
    issues = []
    
    # Check length consistency
    if len(ocr_probs) != len(lm_probs):
        issues.append(f"OCR and LM probability lengths mismatch: {len(ocr_probs)} vs {len(lm_probs)}")
    
    if len(ocr_probs) != len(gt_probs):
        issues.append(f"OCR and GT probability lengths mismatch: {len(ocr_probs)} vs {len(gt_probs)}")
    
    if len(ocr_probs) != len(topk_indices):
        issues.append(f"Probability vector and topk_indices length mismatch: {len(ocr_probs)} vs {len(topk_indices)}")
    
    # Check sum of GT probabilities
    gt_sum = gt_probs.sum()
    if not np.isclose(gt_sum, 1.0, atol=1e-6):
        issues.append(f"GT probability sum is not 1.0: {gt_sum:.6f}")
    
    # Check if only one GT value is 1
    gt_nonzero = np.count_nonzero(gt_probs > 0.5)
    if gt_nonzero != 1:
        issues.append(f"GT should have exactly one 1.0 value, but found {gt_nonzero} non-zero values")
    
    if issues:
        return False, "\n".join(issues)
    else:
        return True, "Data consistency check passed ✅"


def display_sample(ocr_probs, lm_probs, gt_probs, topk_indices, sample_idx, ocr_chars=None, top_k=10):
    """
    Display detailed information of a single sample
    
    Explanation:
    - topk_indices: Defines the 100 candidate characters (OCR's top-100)
    - ocr_probs:    OCR's scores for these 100 candidates
    - lm_probs:     LM's scores for the same 100 candidates (one-to-one correspondence)
    - gt_probs:     One-hot encoding of the ground truth
    """
    print("\n" + "="*80)
    print(f"📄 Character Sample #{sample_idx}")
    print("="*80)
    
    # Convert to numpy arrays (if tensors)
    if isinstance(ocr_probs, torch.Tensor):
        ocr_probs = ocr_probs.cpu().numpy()
    if isinstance(lm_probs, torch.Tensor):
        lm_probs = lm_probs.cpu().numpy()
    if isinstance(gt_probs, torch.Tensor):
        gt_probs = gt_probs.cpu().numpy()
    if isinstance(topk_indices, torch.Tensor):
        topk_indices = topk_indices.cpu().numpy()
    
    # Remove extra dimensions [1, 100] -> [100]
    ocr_probs = ocr_probs.squeeze()
    lm_probs = lm_probs.squeeze()
    gt_probs = gt_probs.squeeze()
    topk_indices = topk_indices.squeeze()
    
    # Verify data consistency
    is_consistent, consistency_msg = verify_data_consistency(ocr_probs, lm_probs, gt_probs, topk_indices)
    
    # Basic information
    print(f"\n📊 Probability Vector Information:")
    print(f"  ├─ Length: {len(ocr_probs)} (OCR's top-100 candidates)")
    print(f"  ├─ OCR Probability Sum: {ocr_probs.sum():.6f}")
    print(f"  ├─ LM  Probability Sum: {lm_probs.sum():.6f} (for the same 100 candidates)")
    print(f"  ├─ GT  Probability Sum: {gt_probs.sum():.6f} (should be 1.0)")
    print(f"  └─ Consistency Check: {consistency_msg}")
    
    if not is_consistent:
        print(f"\n⚠️ Warning: Data has consistency issues!")
    
    # Find the true OCR index of the GT character
    # gt_probs is one-hot, find the position with value 1.0
    gt_pos_in_topk = np.argmax(gt_probs)  # Position in top-100 list (0-99)
    gt_ocr_idx = topk_indices[gt_pos_in_topk]  # True OCR index (1-based)
    
    # Convert OCR index to character (OCR index is 1-based, so subtract 1)
    gt_char = ocr_chars[gt_ocr_idx - 1] if ocr_chars and 0 < gt_ocr_idx <= len(ocr_chars) else f"Index_{gt_ocr_idx}"
    print(f"\n🎯 Ground Truth (GT) Character: '{gt_char}' (Index: {gt_ocr_idx})")
    
    # OCR prediction (highest probability candidate)
    ocr_top1_pos = np.argmax(ocr_probs)  # Position in top-100
    ocr_top1_ocr_idx = topk_indices[ocr_top1_pos]  # True OCR index
    ocr_top1_char = ocr_chars[ocr_top1_ocr_idx - 1] if ocr_chars and 0 < ocr_top1_ocr_idx <= len(ocr_chars) else f"Index_{ocr_top1_ocr_idx}"
    ocr_top1_prob = ocr_probs[ocr_top1_pos]
    is_correct = (ocr_top1_ocr_idx == gt_ocr_idx)
    status = "🟢 Correct" if is_correct else "🔴 Incorrect"
    print(f"OCR Top-1: '{ocr_top1_char}' (Probability: {ocr_top1_prob:.6f}) {status}")
    
    # LM prediction
    lm_top1_pos = np.argmax(lm_probs)
    lm_top1_ocr_idx = topk_indices[lm_top1_pos]
    lm_top1_char = ocr_chars[lm_top1_ocr_idx - 1] if ocr_chars and 0 < lm_top1_ocr_idx <= len(ocr_chars) else f"Index_{lm_top1_ocr_idx}"
    lm_top1_prob = lm_probs[lm_top1_pos]
    lm_correct = (lm_top1_ocr_idx == gt_ocr_idx)
    lm_status = "🟢 Correct" if lm_correct else "🔴 Incorrect"
    print(f"LM  Top-1: '{lm_top1_char}' (Probability: {lm_top1_prob:.6f}) {lm_status}")
    
    # Rank of GT in OCR
    ocr_sorted_positions = np.argsort(ocr_probs)[::-1]  # Descending order, returns positions in top-100
    gt_rank_in_ocr = np.where(ocr_sorted_positions == gt_pos_in_topk)[0][0] + 1
    print(f"GT Rank in OCR: {gt_rank_in_ocr} (Probability: {ocr_probs[gt_pos_in_topk]:.6f})")
    
    # Rank of GT in LM
    lm_sorted_positions = np.argsort(lm_probs)[::-1]
    gt_rank_in_lm = np.where(lm_sorted_positions == gt_pos_in_topk)[0][0] + 1
    print(f"GT Rank in LM: {gt_rank_in_lm} (Probability: {lm_probs[gt_pos_in_topk]:.6f})")
    
    # Display Top-K candidates (keep OCR's original order, no re-sorting)
    print(f"\n📊 OCR Top-{top_k} Candidates (in OCR's original top-100 order):")
    print(f"Explanation: First {top_k} candidates from OCR, showing OCR and LM scores for the same candidate")
    print(f"Important: Order is consistent for easy comparison of probability differences between OCR and LM")
    print(f"{'Pos':<6} {'Char':<8} {'Index':<8} {'OCR Prob':<12} {'LM Prob':<12} {'Prob Ratio':<10} {'GT':<4}")
    print("-" * 80)
    
    for i in range(min(top_k, len(ocr_probs))):
        # Directly use OCR's top-100 order (i is the position)
        ocr_idx = topk_indices[i]
        char = ocr_chars[ocr_idx - 1] if ocr_chars and 0 < ocr_idx <= len(ocr_chars) else f"idx_{ocr_idx}"
        is_gt = "✅" if ocr_idx == gt_ocr_idx else ""
        
        # Calculate LM/OCR probability ratio to see if LM favors this candidate more
        prob_ratio = lm_probs[i] / ocr_probs[i] if ocr_probs[i] > 1e-10 else 0.0
        ratio_str = f"{prob_ratio:.2f}x" if prob_ratio > 0 else "N/A"
        
        print(f"{i+1:<6} {char:<8} {ocr_idx:<8} {ocr_probs[i]:<12.6f} {lm_probs[i]:<12.6f} {ratio_str:<10} {is_gt:<4}")
    
    print(f"\n💡 Interpretation:")
    print(f"  - 'Pos': Rank in OCR's top-100 (1 = OCR's most favored candidate)")
    print(f"  - 'OCR Prob': OCR's score for this candidate")
    print(f"  - 'LM Prob': LM's score for **the same candidate**")
    print(f"  - 'Prob Ratio': LM Probability / OCR Probability (>1 means LM favors more, <1 means OCR favors more)")
    
    print("="*80)


def display_statistics(data):
    """
    Display overall statistical information of the dataset
    
    Explanation:
    - OCR Accuracy: Whether the top-ranked candidate in OCR's 100 scores is the GT
    - LM Accuracy: Whether the top-ranked candidate in LM's re-scored 100 candidates is the GT
    """
    print("\n" + "="*80)
    print("📈 Overall Dataset Statistics")
    print("="*80)
    print("\nExplanation: OCR and LM score the same candidate characters, only from different sources")
    
    ocr_tensor = data['ocr_c100']
    lm_tensor = data['lm_c100']
    gt_tensor = data['gt_c100']
    
    # Convert to numpy
    ocr_np = ocr_tensor.cpu().numpy().squeeze()
    lm_np = lm_tensor.cpu().numpy().squeeze()
    gt_np = gt_tensor.cpu().numpy().squeeze()
    
    total_samples = ocr_np.shape[0]
    
    # Positions of GT (in top-100)
    gt_positions = np.argmax(gt_np, axis=1)
    
    # OCR Top-1 Accuracy (compare positions in top-100)
    ocr_top1_positions = np.argmax(ocr_np, axis=1)
    ocr_correct = (ocr_top1_positions == gt_positions).sum()
    ocr_accuracy = ocr_correct / total_samples * 100
    
    # LM Top-1 Accuracy
    lm_top1_positions = np.argmax(lm_np, axis=1)
    lm_correct = (lm_top1_positions == gt_positions).sum()
    lm_accuracy = lm_correct / total_samples * 100
    
    print(f"\n【Prediction Accuracy】")
    print(f"  ├─ OCR Top-1 Accuracy: {ocr_accuracy:.2f}% ({ocr_correct:,}/{total_samples:,})")
    print(f"  │   (Accuracy of the top-ranked candidate in OCR's 100 scores)")
    print(f"  └─ LM  Top-1 Accuracy: {lm_accuracy:.2f}% ({lm_correct:,}/{total_samples:,})")
    print(f"      (Accuracy of the top-ranked candidate in LM's re-scored 100 candidates)")
    
    # Probability sum statistics
    ocr_sums = ocr_np.sum(axis=1)
    lm_sums = lm_np.sum(axis=1)
    
    print(f"\n【Probability Sum Statistics】")
    print(f"  OCR Probability Sum:")
    print(f"    ├─ Mean: {ocr_sums.mean():.6f}")
    print(f"    ├─ Min: {ocr_sums.min():.6f}")
    print(f"    ├─ Max: {ocr_sums.max():.6f}")
    print(f"    └─ Std: {ocr_sums.std():.6f}")
    
    print(f"  LM Probability Sum:")
    print(f"    ├─ Mean: {lm_sums.mean():.6f}")
    print(f"    ├─ Min: {lm_sums.min():.6f}")
    print(f"    ├─ Max: {lm_sums.max():.6f}")
    print(f"    └─ Std: {lm_sums.std():.6f}")
    
    # Samples correctly predicted by both OCR and LM
    both_correct = ((ocr_top1_positions == gt_positions) & (lm_top1_positions == gt_positions)).sum()
    print(f"\n【Consistency Analysis】")
    print(f"  ├─ Correctly Predicted by Both OCR and LM: {both_correct:,} ({both_correct/total_samples*100:.2f}%)")
    print(f"  ├─ Correctly Predicted Only by OCR: {(ocr_correct - both_correct):,}")
    print(f"  ├─ Correctly Predicted Only by LM: {(lm_correct - both_correct):,}")
    print(f"  └─ Incorrectly Predicted by Both: {(total_samples - ocr_correct - lm_correct + both_correct):,}")
    
    print("="*80)


def interactive_mode(data, ocr_chars=None):
    """Interactive viewing mode"""
    ocr_tensor = data['ocr_c100']
    lm_tensor = data['lm_c100']
    gt_tensor = data['gt_c100']
    topk_indices_tensor = data.get('topk_indices', None)
    total_samples = ocr_tensor.shape[0]
    
    # Check if topk_indices exists
    if topk_indices_tensor is None:
        print("⚠️ Warning: 'topk_indices' not found in data, cannot display characters correctly!")
        print("Please re-generate data with the latest version of prepare_fusion_data.py.\n")
    
    print("\n" + "="*80)
    print("🔍 Interactive Viewing Mode")
    print("="*80)
    print(f"Total Samples: {total_samples:,}")
    print("\nCommands:")
    print(f"  - Enter a number: View sample at the specified index (0-{total_samples-1})")
    print("  - 'r' or 'random': View a random sample")
    print("  - 's' or 'stats': Display overall statistics")
    print("  - 'q' or 'quit': Exit")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input in ['s', 'stats']:
                display_statistics(data)
            
            elif user_input in ['r', 'random']:
                import random
                sample_idx = random.randint(0, total_samples - 1)
                print(f"\n🎲 Randomly selected sample #{sample_idx}")
                if topk_indices_tensor is not None:
                    display_sample(
                        ocr_tensor[sample_idx],
                        lm_tensor[sample_idx],
                        gt_tensor[sample_idx],
                        topk_indices_tensor[sample_idx],
                        sample_idx,
                        ocr_chars
                    )
                else:
                    print("❌ Cannot display sample: Missing topk_indices data")
            
            else:
                # Try to parse as index
                try:
                    sample_idx = int(user_input)
                    if 0 <= sample_idx < total_samples:
                        if topk_indices_tensor is not None:
                            display_sample(
                                ocr_tensor[sample_idx],
                                lm_tensor[sample_idx],
                                gt_tensor[sample_idx],
                                topk_indices_tensor[sample_idx],
                                sample_idx,
                                ocr_chars
                            )
                        else:
                            print("❌ Cannot display sample: Missing topk_indices data")
                    else:
                        print(f"❌ Index out of range! Please enter a number between 0 and {total_samples-1}.")
                except ValueError:
                    print("❌ Invalid command! Please enter a number, 'r'(random), 's'(stats), or 'q'(quit).")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error occurred: {e}")


def main():
    """
    Main function
    
    Features:
    1. Load and display fusion training data
    2. Verify one-to-one correspondence between OCR_100 and LM_100
    3. Interactively view sample data
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("="*80)
        print("📖 Fusion Training Data Viewer")
        print("="*80)
        print("\nCore Concepts:")
        print("  - OCR_100: OCR model's top-100 candidates and their probabilities")
        print("  - LM_100:  Probabilities of the same 100 candidates in LM (one-to-one correspondence)")
        print("  - GT_100:  One-hot encoding of the ground truth character")
        print("\nUsage: python view_pt_data.py <pt_file> [ocr_char_file]")
        print("\nExamples:")
        print("  python view_pt_data.py fusion_training_data.pt")
        print("  python view_pt_data.py fusion_training_data.pt ppocr_keys_v1.txt")
        print("="*80)
        sys.exit(1)
    
    pt_file = sys.argv[1]
    ocr_char_file = sys.argv[2] if len(sys.argv) > 2 else 'ppocr_keys_v1.txt'
    
    # Check if file exists
    if not os.path.exists(pt_file):
        print(f"❌ Error: File not found - {pt_file}")
        sys.exit(1)
    
    # Load data
    data = load_pt_file(pt_file)
    
    # Display file information
    display_file_info(data)
    
    # Load OCR character list
    ocr_chars = load_ocr_chars(ocr_char_file)
    if ocr_chars:
        print(f"✅ OCR character list loaded, total {len(ocr_chars)} characters\n")
    
    # Display overall statistics
    display_statistics(data)
    
    # Enter interactive mode
    interactive_mode(data, ocr_chars)


if __name__ == "__main__":
    main()