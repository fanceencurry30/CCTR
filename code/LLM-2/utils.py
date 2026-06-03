import torch
from config import *

def compute_n95(logits, labels, mask_positions):
    # N95
    mask_logits = logits[mask_positions.bool()]
    mask_labels = labels[mask_positions.bool()]
    sorted_probs, sorted_indices = torch.sort(torch.softmax(mask_logits, dim=-1), descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    n95 = torch.sum(cumulative_probs < 0.95, dim=-1) + 1
    correct_in_top_n = (sorted_indices[:, :n95.max()] == mask_labels.unsqueeze(1)).any(dim=1)
    return n95, correct_in_top_n

def get_top_n_candidates(logits, mask_positions, vocab, n=20):
    # top-n
    mask_logits = logits[mask_positions.bool()]
    top_n_probs, top_n_indices = torch.topk(torch.softmax(mask_logits, dim=-1), k=n, dim=-1)
    candidates = []
    for i in range(mask_logits.size(0)):
        candidate_chars = [vocab.idx_to_char[idx.item()] for idx in top_n_indices[i]]
        candidate_probs = top_n_probs[i].tolist()
        candidates.append((candidate_chars, candidate_probs))
    return candidates