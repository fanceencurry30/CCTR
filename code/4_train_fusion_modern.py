"""
🚀 Fusion Layer Training Script - Modern Version

Based on state-of-the-art deep learning best practices:
1. ✅ Cosine Annealing with Warmup (Same as BERT/GPT/ViT)
2. ✅ AdamW Optimizer (Decoupled Weight Decay)
3. ✅ Label Smoothing (Prevent overfitting)
4. ✅ Gradient Clipping (Training stability)
5. ✅ Mixed Precision Training (AMP, accelerate training)
6. ✅ Gradient Accumulation (Support larger effective batch size)
7. ✅ EMA (Exponential Moving Average, improve generalization)

References:
- "Attention Is All You Need" (Transformer original paper)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Decoupled Weight Decay Regularization" (AdamW)
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    # PyTorch 2.0+ new API
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch 1.x legacy API
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta, datetime
import glob
import argparse
from collections import defaultdict
import copy
import json
import csv

from fusion_model_clean import CrossAttentionFusion
from config_fusion import *


# ============================================
# 📝 Training Logger
# ============================================

class TrainingLogger:
    """
    Training Logger
    
    Features:
    1. Record detailed training information for each epoch
    2. Save in JSON and CSV formats
    3. Real-time update, no data loss even if training is interrupted
    """
    
    def __init__(self, log_dir, experiment_name=None):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Experiment name (uses timestamp by default)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"fusion_training_{timestamp}"
        
        self.experiment_name = experiment_name
        
        # Log file paths
        self.json_log_path = os.path.join(log_dir, f"{experiment_name}.json")
        self.csv_log_path = os.path.join(log_dir, f"{experiment_name}.csv")
        self.summary_path = os.path.join(log_dir, f"{experiment_name}_summary.txt")
        
        # Training history
        self.history = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'epochs': []
        }
        
        # CSV headers
        self.csv_headers = [
            'epoch', 'train_loss', 'val_acc', 'learning_rate',
            'train_ocr_lm_both_correct', 'train_ocr_only_correct', 
            'train_lm_only_correct', 'train_both_wrong',
            'val_ocr_lm_both_acc', 'val_ocr_only_acc', 
            'val_lm_only_acc', 'val_both_wrong_acc',
            'val_ocr_lm_both_count', 'val_ocr_only_count',
            'val_lm_only_count', 'val_both_wrong_count',
            'is_best_model'
        ]
        
        print(f"\n📝 Training logger initialized")
        print(f"  ├─ Experiment name: {experiment_name}")
        print(f"  ├─ JSON log: {self.json_log_path}")
        print(f"  ├─ CSV log: {self.csv_log_path}")
        print(f"  └─ Summary file: {self.summary_path}")
    
    def log_config(self, config_dict):
        """Record training configuration"""
        self.history['config'] = config_dict
        self._save_json()
    
    def log_epoch(self, epoch_data, is_best=False):
        """
        Record training data for one epoch
        
        Args:
            epoch_data: Dictionary containing training information
            is_best: Whether it's the best model
        """
        epoch_data['is_best_model'] = is_best
        epoch_data['timestamp'] = datetime.now().isoformat()
        
        self.history['epochs'].append(epoch_data)
        
        # Save JSON
        self._save_json()
        
        # Append to CSV
        self._append_csv(epoch_data)
        
        # Update summary
        self._update_summary()
    
    def _save_json(self):
        """Save JSON log"""
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _append_csv(self, epoch_data):
        """Append CSV row"""
        file_exists = os.path.exists(self.csv_log_path)
        
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            # Extract fields needed for CSV
            csv_row = {}
            for key in self.csv_headers:
                csv_row[key] = epoch_data.get(key, 'N/A')
            
            writer.writerow(csv_row)
    
    def _update_summary(self):
        """Update training summary"""
        if not self.history['epochs']:
            return
        
        # Find best epoch
        best_epoch = max(self.history['epochs'], key=lambda x: x.get('val_acc', 0))
        latest_epoch = self.history['epochs'][-1]
        
        summary_lines = [
            "="*80,
            f"📊 Training Summary - {self.experiment_name}",
            "="*80,
            "",
            f"🕐 Start time: {self.history['start_time']}",
            f"🕑 Last update: {latest_epoch['timestamp']}",
            f"📈 Total epochs: {len(self.history['epochs'])}",
            "",
            "="*80,
            "🏆 Best Model",
            "="*80,
            f"  ├─ Epoch: {best_epoch['epoch']}",
            f"  ├─ Validation Accuracy: {best_epoch['val_acc']:.4f}",
            f"  ├─ Training Loss: {best_epoch['train_loss']:.4f}",
            f"  └─ Learning Rate: {best_epoch['learning_rate']:.6e}",
            "",
            "="*80,
            "📊 Best Model - Validation Set Accuracy by Sample Type",
            "="*80,
            f"  ├─ OCR✅ LM✅: {best_epoch.get('val_ocr_lm_both_acc', 0):.2f}% ({best_epoch.get('val_ocr_lm_both_count', 0)} samples)",
            f"  ├─ OCR✅ LM❌: {best_epoch.get('val_ocr_only_acc', 0):.2f}% ({best_epoch.get('val_ocr_only_count', 0)} samples)",
            f"  ├─ OCR❌ LM✅: {best_epoch.get('val_lm_only_acc', 0):.2f}% ({best_epoch.get('val_lm_only_count', 0)} samples) ⭐",
            f"  └─ OCR❌ LM❌: {best_epoch.get('val_both_wrong_acc', 0):.2f}% ({best_epoch.get('val_both_wrong_count', 0)} samples) 🎯",
            "",
            "="*80,
            "📈 Latest Epoch",
            "="*80,
            f"  ├─ Epoch: {latest_epoch['epoch']}",
            f"  ├─ Validation Accuracy: {latest_epoch['val_acc']:.4f}",
            f"  ├─ Training Loss: {latest_epoch['train_loss']:.4f}",
            f"  └─ Learning Rate: {latest_epoch['learning_rate']:.6e}",
            "",
        ]
        
        # Add training configuration
        if self.history['config']:
            summary_lines.extend([
                "="*80,
                "⚙️ Training Configuration",
                "="*80,
            ])
            for key, value in self.history['config'].items():
                summary_lines.append(f"  ├─ {key}: {value}")
            summary_lines.append("")
        
        # Add progress curve (last 10 epochs)
        recent_epochs = self.history['epochs'][-10:]
        summary_lines.extend([
            "="*80,
            "📉 Training Curve (Last 10 Epochs)",
            "="*80,
        ])
        
        summary_lines.append(f"{'Epoch':<8} {'Loss':<10} {'Val Acc':<10} {'LR':<12} {'Best':<6}")
        summary_lines.append("-"*80)
        
        for ep in recent_epochs:
            best_mark = "⭐" if ep.get('is_best_model', False) else ""
            summary_lines.append(
                f"{ep['epoch']:<8} "
                f"{ep['train_loss']:<10.4f} "
                f"{ep['val_acc']:<10.4f} "
                f"{ep['learning_rate']:<12.6e} "
                f"{best_mark:<6}"
            )
        
        summary_lines.append("")
        summary_lines.append("="*80)
        
        # Write to file
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))


# ============================================
# 📊 Dataset Class
# ============================================

class MultiPTDataset(Dataset):
    """Multi-PT File Dataset"""
    
    def __init__(self, pt_file_paths, rank=0):
        if rank == 0:
            print(f"\n{'='*80}")
            print("📦 Loading dataset...")
            print(f"{'='*80}")
        
        all_ocr = []
        all_lm = []
        all_gt = []
        
        for pt_file in pt_file_paths:
            if rank == 0:
                print(f"\nLoading: {os.path.basename(pt_file)}")
            
            data = torch.load(pt_file, map_location='cpu')
            
            ocr_c100 = data['ocr_c100']
            lm_c100 = data['lm_c100']
            gt_c100 = data['gt_c100']
            
            # Ensure dimension [N, 1, 100]
            if ocr_c100.dim() == 2:
                ocr_c100 = ocr_c100.unsqueeze(1)
            if lm_c100.dim() == 2:
                lm_c100 = lm_c100.unsqueeze(1)
            if gt_c100.dim() == 2:
                gt_c100 = gt_c100.unsqueeze(1)
            
            all_ocr.append(ocr_c100)
            all_lm.append(lm_c100)
            all_gt.append(gt_c100)
        
        # Concatenate all data
        self.ocr_c100 = torch.cat(all_ocr, dim=0)
        self.lm_c100 = torch.cat(all_lm, dim=0)
        self.gt_c100 = torch.cat(all_gt, dim=0)
        
        if rank == 0:
            print(f"\n✅ Dataset loaded successfully")
            print(f"  └─ Total samples: {len(self):,}")
    
    def __len__(self):
        return len(self.ocr_c100)
    
    def __getitem__(self, idx):
        return {
            'ocr': self.ocr_c100[idx],
            'lm': self.lm_c100[idx],
            'gt': self.gt_c100[idx]
        }


# ============================================
# 🎯 Learning Rate Scheduler: Cosine Annealing with Warmup
# ============================================

class CosineWarmupScheduler:
    """
    Cosine Annealing with Warmup
    
    Widely used in:
    - BERT (Google, 2018)
    - GPT-2/3 (OpenAI)
    - Vision Transformer (Google, 2020)
    - CLIP (OpenAI, 2021)
    
    Advantages:
    1. Warmup phase: Prevent gradient explosion in early training
    2. Cosine decay: Smooth and natural learning rate decrease
    3. No manual adjustment: Only need to set total epochs and warmup epochs
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 max_lr, min_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        
        # Initialize learning rate
        self.step(0)
    
    def get_lr(self, epoch):
        """Calculate learning rate for current epoch"""
        if epoch < self.warmup_epochs:
            # Warmup phase: Linear increase
            return self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def step(self, epoch):
        """Update learning rate"""
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.last_epoch = epoch
        return lr
    
    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
        }
    
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']


# ============================================
# 🔧 Exponential Moving Average (EMA)
# ============================================

class EMA:
    """
    Exponential Moving Average
    
    Used to smooth model parameters and improve generalization
    Applied in: YOLO, Stable Diffusion, etc.
    """
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters (for validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters (after validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================
# 📉 Label Smoothing Loss
# ============================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy
    
    Prevent model overconfidence and improve generalization
    Paper: "Rethinking the Inception Architecture for Computer Vision"
    """
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch, num_classes] or [batch, seq_len, num_classes] model prediction logits
            target: [batch] or [batch, seq_len] class indices, or [batch, num_classes] one-hot
        """
        # Handle different input dimensions
        if pred.dim() == 3:
            # Sequence model output: [batch, seq_len, num_classes]
            batch_size, seq_len, num_classes = pred.shape
            pred = pred.reshape(-1, num_classes)  # [batch*seq_len, num_classes]
            
            # Convert target to class indices if one-hot
            if target.dim() == 3:
                target = target.argmax(dim=-1)  # [batch, seq_len]
            target = target.reshape(-1)  # [batch*seq_len]
            
        elif pred.dim() == 2:
            # Classification model output: [batch, num_classes]
            num_classes = pred.size(1)
            
            # Convert target to class indices if one-hot
            if target.dim() == 2:
                target = target.argmax(dim=-1)  # [batch]
        else:
            raise ValueError(f"pred dimension should be 2 or 3, but got {pred.dim()}")
        
        # Ensure target is int64 type and correct dimension
        target = target.long()
        
        # Ensure pred is 2D [N, num_classes]
        if pred.dim() != 2:
            raise ValueError(f"pred should be 2D after processing, but got {pred.dim()}D, shape={pred.shape}")
        
        # Ensure target is 1D [N]
        if target.dim() != 1:
            raise ValueError(f"target should be 1D after processing, but got {target.dim()}D, shape={target.shape}")
        
        # Log softmax
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# ============================================
# 🏋️ Training Function
# ============================================

def train_epoch(model, train_loader, optimizer, criterion, device, rank, 
                scaler, ema=None, grad_accum_steps=1, epoch=0, world_size=1):
    """
    Train one epoch (supports multi-GPU)
    
    Args:
        grad_accum_steps: Gradient accumulation steps (simulate larger batch size)
        world_size: Number of GPUs
    
    Returns:
        avg_loss: Average loss
        train_stats: Training set sample distribution statistics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Count training set sample distribution
    train_stats = {
        'ocr_correct_lm_correct': 0,
        'ocr_correct_lm_wrong': 0,
        'ocr_wrong_lm_correct': 0,
        'ocr_wrong_lm_wrong': 0,
        'total_samples': 0
    }
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    else:
        pbar = train_loader
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        ocr = batch['ocr'].to(device)  # [batch, 1, 100]
        lm = batch['lm'].to(device)
        gt = batch['gt'].to(device)
        
        # Count OCR and LM predictions
        gt_idx = gt.squeeze(1).argmax(dim=-1)  # [batch]
        ocr_pred = ocr.squeeze(1).argmax(dim=-1)  # [batch]
        lm_pred = lm.squeeze(1).argmax(dim=-1)   # [batch]
        
        ocr_correct = (ocr_pred == gt_idx)
        lm_correct = (lm_pred == gt_idx)
        
        # Count sample types
        train_stats['ocr_correct_lm_correct'] += (ocr_correct & lm_correct).sum().item()
        train_stats['ocr_correct_lm_wrong'] += (ocr_correct & (~lm_correct)).sum().item()
        train_stats['ocr_wrong_lm_correct'] += ((~ocr_correct) & lm_correct).sum().item()
        train_stats['ocr_wrong_lm_wrong'] += ((~ocr_correct) & (~lm_correct)).sum().item()
        train_stats['total_samples'] += gt_idx.size(0)
        
        # Mixed Precision Training
        try:
            # PyTorch 2.0+ new API
            with autocast('cuda'):
                output = model(ocr, lm)  # [batch, 100] - model outputs logits (not probabilities)
                # Convert gt from 3D [batch, 1, 100] to 2D [batch, 100] one-hot encoding
                # Criterion will convert one-hot to class indices internally, then calculate label smoothing CE
                gt_squeezed = gt.squeeze(1)  # [batch, 100]
                loss = criterion(output, gt_squeezed)
                loss = loss / grad_accum_steps  # Gradient accumulation
        except TypeError:
            # PyTorch 1.x legacy API (compatibility)
            with autocast():
                output = model(ocr, lm)  # [batch, 100] - logits
                gt_squeezed = gt.squeeze(1)  # [batch, 100]
                loss = criterion(output, gt_squeezed)
                loss = loss / grad_accum_steps  # Gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation: update every grad_accum_steps steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # EMA update
            if ema is not None:
                ema.update()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
    
    # 🔥 Key: In DDP, loss is calculated based on actual processed samples
    # No need to sync avg_loss as DDP already syncs gradients automatically
    # But you can choose to sync total_loss and num_batches for accurate loss printing
    avg_loss = total_loss / num_batches
    
    return avg_loss, train_stats


# ============================================
# 🔄 Multi-GPU Synchronization Functions
# ============================================

def sync_across_gpus(value, world_size):
    """
    Synchronize values across all GPUs (sum)
    
    Args:
        value: Single value or tensor
        world_size: Number of GPUs
    
    Returns:
        Synchronized value
    """
    if world_size == 1:
        return value
    
    # Convert to tensor
    if not isinstance(value, torch.Tensor):
        value_tensor = torch.tensor(value, dtype=torch.float32).cuda()
    else:
        value_tensor = value.cuda()
    
    # Sum across GPUs
    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    
    return value_tensor.item()


def sync_stats_across_gpus(stats_dict, world_size):
    """
    Synchronize all values in statistics dictionary
    
    Args:
        stats_dict: Dictionary containing statistical data
        world_size: Number of GPUs
    
    Returns:
        Synchronized dictionary
    """
    if world_size == 1:
        return stats_dict
    
    synced_stats = {}
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            # Recursively sync nested dictionaries
            synced_stats[key] = sync_stats_across_gpus(value, world_size)
        elif isinstance(value, (int, float)):
            # Sync numerical values
            synced_stats[key] = sync_across_gpus(value, world_size)
        else:
            # Keep other types unchanged
            synced_stats[key] = value
    
    return synced_stats


# ============================================
# 📊 Evaluation Function
# ============================================

@torch.no_grad()
def evaluate(model, val_loader, device, rank, world_size, ema=None):
    """
    Evaluate model (supports multi-GPU)
    
    Args:
        world_size: Number of GPUs, used for statistics synchronization
    
    Returns:
        accuracy: Overall accuracy
        val_stats: Detailed statistics of validation set samples by type
    """
    
    # Switch to EMA parameters if used
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    # Detailed statistics of performance by sample type
    val_stats = {
        'ocr_correct_lm_correct': {'total': 0, 'fusion_correct': 0},
        'ocr_correct_lm_wrong': {'total': 0, 'fusion_correct': 0},
        'ocr_wrong_lm_correct': {'total': 0, 'fusion_correct': 0},
        'ocr_wrong_lm_wrong': {'total': 0, 'fusion_correct': 0}
    }
    
    if rank == 0:
        pbar = tqdm(val_loader, desc="Evaluating")
    else:
        pbar = val_loader
    
    for batch in pbar:
        ocr = batch['ocr'].to(device)
        lm = batch['lm'].to(device)
        gt = batch['gt'].to(device)
        
        try:
            # PyTorch 2.0+ new API
            with autocast('cuda'):
                output = model(ocr, lm)  # [batch, 100] - logits
        except TypeError:
            # PyTorch 1.x legacy API (compatibility)
            with autocast():
                output = model(ocr, lm)  # [batch, 100] - logits
        
        # Calculate accuracy (argmax of logits = argmax of probabilities)
        pred_idx = output.argmax(dim=-1)  # [batch]
        gt_idx = gt.squeeze(1).argmax(dim=-1)        # [batch]
        ocr_pred = ocr.squeeze(1).argmax(dim=-1)     # [batch]
        lm_pred = lm.squeeze(1).argmax(dim=-1)       # [batch]
        
        fusion_correct = (pred_idx == gt_idx)
        ocr_correct = (ocr_pred == gt_idx)
        lm_correct = (lm_pred == gt_idx)
        
        # Count performance by sample type
        mask_ocr_right_lm_right = ocr_correct & lm_correct
        mask_ocr_right_lm_wrong = ocr_correct & (~lm_correct)
        mask_ocr_wrong_lm_right = (~ocr_correct) & lm_correct
        mask_ocr_wrong_lm_wrong = (~ocr_correct) & (~lm_correct)
        
        val_stats['ocr_correct_lm_correct']['total'] += mask_ocr_right_lm_right.sum().item()
        val_stats['ocr_correct_lm_correct']['fusion_correct'] += (fusion_correct & mask_ocr_right_lm_right).sum().item()
        
        val_stats['ocr_correct_lm_wrong']['total'] += mask_ocr_right_lm_wrong.sum().item()
        val_stats['ocr_correct_lm_wrong']['fusion_correct'] += (fusion_correct & mask_ocr_right_lm_wrong).sum().item()
        
        val_stats['ocr_wrong_lm_correct']['total'] += mask_ocr_wrong_lm_right.sum().item()
        val_stats['ocr_wrong_lm_correct']['fusion_correct'] += (fusion_correct & mask_ocr_wrong_lm_right).sum().item()
        
        val_stats['ocr_wrong_lm_wrong']['total'] += mask_ocr_wrong_lm_wrong.sum().item()
        val_stats['ocr_wrong_lm_wrong']['fusion_correct'] += (fusion_correct & mask_ocr_wrong_lm_wrong).sum().item()
        
        total_correct += fusion_correct.sum().item()
        total_samples += gt_idx.size(0)
        
        if rank == 0:
            pbar.set_postfix({'acc': f'{total_correct/total_samples:.4f}'})
    
    # 🔥 Key: Synchronize all statistics across GPUs
    total_correct = sync_across_gpus(total_correct, world_size)
    total_samples = sync_across_gpus(total_samples, world_size)
    val_stats = sync_stats_across_gpus(val_stats, world_size)
    
    accuracy = total_correct / total_samples
    
    # Restore original parameters
    if ema is not None:
        ema.restore()
    
    return accuracy, val_stats


# ============================================
# 🚀 Main Training Pipeline
# ============================================

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                           timeout=timedelta(seconds=3600))

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train(rank, world_size, args):
    """Main training function"""
    
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # ============================================
    # 0. Initialize Training Logger (main process only)
    # ============================================
    
    logger = None
    if rank == 0:
        log_dir = os.path.join(os.path.dirname(FUSION_MODEL_SAVE_PATH), "logs")
        logger = TrainingLogger(log_dir, experiment_name=args.experiment_name)
    
    # ============================================
    # 1. Load Data
    # ============================================
    
    # Scan PT files
    pattern = os.path.join(args.data_dir, args.pattern)
    pt_files = sorted(glob.glob(pattern))
    
    if rank == 0:
        print(f"\n🔍 Found {len(pt_files)} data files:")
        for f in pt_files:
            print(f"  - {os.path.basename(f)}")
    
    # Create dataset
    full_dataset = MultiPTDataset(pt_files, rank=rank)
    
    # Split into training and validation sets
    total_size = len(full_dataset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_size))
    
    if rank == 0:
        print(f"\n📊 Dataset Split:")
        print(f"  ├─ Training set: {train_size:,} samples")
        print(f"  └─ Validation set: {val_size:,} samples")
    
    # Create DataLoaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # ============================================
    # 2. Create Model
    # ============================================
    
    model = CrossAttentionFusion(
        feature_dim=FEATURE_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # ============================================
    # 3. Optimizer and Scheduler
    # ============================================
    
    # AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,  # Will be overwritten by scheduler
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    # Cosine Warmup scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        max_lr=LEARNING_RATE,
        min_lr=MIN_LEARNING_RATE
    )
    
    # Label Smoothing loss
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    
    # Mixed Precision Scaler
    try:
        # PyTorch 2.0+ new API
        scaler = GradScaler('cuda')
    except TypeError:
        # PyTorch 1.x legacy API (compatibility)
        scaler = GradScaler()
    
    # EMA (optional, recommended)
    ema = EMA(model, decay=0.9999) if args.use_ema else None
    
    if rank == 0:
        print(f"\n🎯 Training Configuration:")
        print(f"  ├─ Optimizer: AdamW")
        print(f"  ├─ Initial Learning Rate: {LEARNING_RATE:.2e}")
        print(f"  ├─ Minimum Learning Rate: {MIN_LEARNING_RATE:.2e}")
        print(f"  ├─ Warmup Epochs: {WARMUP_EPOCHS}")
        print(f"  ├─ Weight Decay: {WEIGHT_DECAY}")
        print(f"  ├─ Label Smoothing: {LABEL_SMOOTHING}")
        print(f"  ├─ Gradient Clipping: {GRAD_CLIP}")
        print(f"  ├─ Batch Size: {BATCH_SIZE}")
        print(f"  ├─ Mixed Precision: ✅ Enabled")
        print(f"  └─ EMA: {'✅ Enabled' if ema else '❌ Disabled'}")
        
        # Record configuration to log
        if logger is not None:
            config_dict = {
                'optimizer': 'AdamW',
                'learning_rate': LEARNING_RATE,
                'min_learning_rate': MIN_LEARNING_RATE,
                'warmup_epochs': WARMUP_EPOCHS,
                'total_epochs': EPOCHS,
                'weight_decay': WEIGHT_DECAY,
                'label_smoothing': LABEL_SMOOTHING,
                'gradient_clipping': GRAD_CLIP,
                'batch_size': BATCH_SIZE,
                'num_gpus': world_size,
                'feature_dim': FEATURE_DIM,
                'num_heads': NUM_HEADS,
                'hidden_dim': HIDDEN_DIM,
                'dropout': DROPOUT,
                'num_encoder_layers': NUM_ENCODER_LAYERS,
                'num_decoder_layers': NUM_DECODER_LAYERS,
                'mixed_precision': True,
                'ema_enabled': args.use_ema,
                'ema_decay': 0.9999 if args.use_ema else None,
                'grad_accum_steps': args.grad_accum_steps,
                'train_samples': train_size,
                'val_samples': val_size,
            }
            logger.log_config(config_dict)
    
    # ============================================
    # 4. Load Checkpoint (if exists)
    # ============================================
    
    best_model_path = f"{FUSION_MODEL_SAVE_PATH}_best.pth"
    checkpoint_path = f"{FUSION_MODEL_SAVE_PATH}_checkpoint.pth"
    
    start_epoch = 0
    best_val_acc = 0.0
    
    if os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"\n📂 Checkpoint found, restoring...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_acc']
        
        if rank == 0:
            print(f"   ✅ Resumed training from Epoch {start_epoch}")
            print(f"   ✅ Best Validation Accuracy: {best_val_acc:.4f}")
    
    # ============================================
    # 5. Training Loop
    # ============================================
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("🚀 Starting Training")
        print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_stats = train_epoch(
            model, train_loader, optimizer, criterion, device, rank,
            scaler, ema, grad_accum_steps=args.grad_accum_steps, epoch=epoch,
            world_size=world_size
        )
        
        # 🔥 Sync training statistics (multi-GPU)
        train_stats = sync_stats_across_gpus(train_stats, world_size)
        
        # Evaluate
        val_acc, val_stats = evaluate(model, val_loader, device, rank, world_size, ema)
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"{'='*80}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Current Learning Rate: {current_lr:.6e}")
            
            # Print training set sample distribution
            print(f"\n📊 Training Set Sample Distribution:")
            total = int(train_stats['total_samples'])
            if total > 0:
                ocr_lm_both = int(train_stats['ocr_correct_lm_correct'])
                ocr_only = int(train_stats['ocr_correct_lm_wrong'])
                lm_only = int(train_stats['ocr_wrong_lm_correct'])
                both_wrong = int(train_stats['ocr_wrong_lm_wrong'])
                
                print(f"  ├─ OCR✅ LM✅: {ocr_lm_both:,} ({ocr_lm_both/total*100:.1f}%)")
                print(f"  ├─ OCR✅ LM❌: {ocr_only:,} ({ocr_only/total*100:.1f}%)")
                print(f"  ├─ OCR❌ LM✅: {lm_only:,} ({lm_only/total*100:.1f}%) ⭐")
                print(f"  └─ OCR❌ LM❌: {both_wrong:,} ({both_wrong/total*100:.1f}%) 🎯")
            
            # Print validation set accuracy by sample type
            print(f"\n📈 Validation Set Accuracy by Sample Type:")
            val_ocr_lm_both_acc = 0
            val_ocr_only_acc = 0
            val_lm_only_acc = 0
            val_both_wrong_acc = 0
            
            for key, label, emoji in [
                ('ocr_correct_lm_correct', 'OCR✅ LM✅', ''),
                ('ocr_correct_lm_wrong', 'OCR✅ LM❌', ''),
                ('ocr_wrong_lm_correct', 'OCR❌ LM✅', '⭐'),
                ('ocr_wrong_lm_wrong', 'OCR❌ LM❌', '🎯')
            ]:
                total_samples = int(val_stats[key]['total'])
                correct_samples = int(val_stats[key]['fusion_correct'])
                if total_samples > 0:
                    acc = correct_samples / total_samples * 100
                    print(f"  ├─ {label}: {acc:.2f}% ({correct_samples}/{total_samples}) {emoji}")
                    
                    # Save accuracy by type
                    if key == 'ocr_correct_lm_correct':
                        val_ocr_lm_both_acc = acc
                    elif key == 'ocr_correct_lm_wrong':
                        val_ocr_only_acc = acc
                    elif key == 'ocr_wrong_lm_correct':
                        val_lm_only_acc = acc
                    elif key == 'ocr_wrong_lm_wrong':
                        val_both_wrong_acc = acc
                else:
                    print(f"  ├─ {label}: N/A (no samples)")
            
            # Determine if it's the best model
            is_best = False
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True
                
                # Save model (use EMA parameters if available)
                if ema is not None:
                    ema.apply_shadow()
                    torch.save(model.module.state_dict(), best_model_path)
                    ema.restore()
                else:
                    torch.save(model.module.state_dict(), best_model_path)
                
                print(f"\n🎉 New best model! Validation Accuracy: {best_val_acc:.4f}")
                print(f"   Saved to: {best_model_path}")
            
            # Record epoch data to log
            if logger is not None:
                epoch_data = {
                    'epoch': epoch + 1,
                    'train_loss': float(train_loss),
                    'val_acc': float(val_acc),
                    'learning_rate': float(current_lr),
                    # Training set sample distribution
                    'train_ocr_lm_both_correct': int(train_stats['ocr_correct_lm_correct']),
                    'train_ocr_only_correct': int(train_stats['ocr_correct_lm_wrong']),
                    'train_lm_only_correct': int(train_stats['ocr_wrong_lm_correct']),
                    'train_both_wrong': int(train_stats['ocr_wrong_lm_wrong']),
                    # Validation set accuracy by sample type
                    'val_ocr_lm_both_acc': float(val_ocr_lm_both_acc),
                    'val_ocr_only_acc': float(val_ocr_only_acc),
                    'val_lm_only_acc': float(val_lm_only_acc),
                    'val_both_wrong_acc': float(val_both_wrong_acc),
                    # Validation set sample count by type
                    'val_ocr_lm_both_count': int(val_stats['ocr_correct_lm_correct']['total']),
                    'val_ocr_only_count': int(val_stats['ocr_correct_lm_wrong']['total']),
                    'val_lm_only_count': int(val_stats['ocr_wrong_lm_correct']['total']),
                    'val_both_wrong_count': int(val_stats['ocr_wrong_lm_wrong']['total']),
                }
                logger.log_epoch(epoch_data, is_best=is_best)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved")
            print(f"{'='*80}\n")
    
    cleanup()


# ============================================
# 🎬 Entry Function
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Fusion Layer Training - Modern Version')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='Data directory')
    parser.add_argument('--pattern', type=str, default='*_normalized.pt',
                       help='Data file matching pattern')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA (recommended)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='Gradient accumulation steps (simulate larger batch size)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (uses timestamp by default)')
    
    args = parser.parse_args()
    
    # Check number of GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No available GPUs detected!")
    
    print(f"\n🎮 Detected {world_size} GPUs")
    
    # Launch distributed training
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()