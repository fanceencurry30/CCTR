import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import os
import argparse
import logging
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerLM
from data_loader import CharVocab, BlankFillingDataset
from config import *
from tqdm import tqdm


def setup_logger(rank, log_dir="logs"):
    """Set up logger"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'train_{timestamp}_rank{rank}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler (only print for rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File log format
    file_formatter = logging.Formatter('[%(asctime)s][Rank %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """Training process running on each GPU"""
    
    # Set up distributed training
    setup(rank, world_size)
    
    # Set up logger
    logger = setup_logger(rank, args.log_dir)
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info("Start Training")
        logger.info("=" * 70)
        logger.info(f"Using {world_size} GPUs: {args.gpus}")
        logger.info(f"Log directory: {args.log_dir}")
    
    # Load vocabulary
    vocab = CharVocab(DICTIONARY_PATH)
    
    if rank == 0:
        logger.info(f"\nLoading data: {TRAIN_DATA_PATH}")
    
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    
    # Create dataset
    full_dataset = BlankFillingDataset(texts, vocab)
    
    # 100% data for training (no validation set)
    train_dataset = full_dataset
    
    if rank == 0:
        logger.info(f"\nDataset Split:")
        logger.info(f"  Training set: {len(train_dataset)} (100%)")
        logger.info(f"  Validation set: 0 (not used)")
    
    # Create distributed Sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    if rank == 0:
        logger.info(f"\nInitializing Model:")
        logger.info(f"  Vocabulary size: {vocab.vocab_size}")
        logger.info(f"  Embedding dimension: {EMBED_DIM}")
        logger.info(f"  Number of layers: {NUM_LAYERS}")
        logger.info(f"  Number of attention heads: {NUM_HEADS}")
    
    model = TransformerLM(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        use_linformer=args.use_linformer,
        k_dim=SPARSE_QUERIES
    )
    
    # Move model to GPU and wrap with DDP
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"\nTotal parameters: {total_params:,}")
        logger.info(f"Using device: GPU {rank} (total {world_size} GPUs)")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Mixed precision training (saves memory, accelerates training)
    scaler = GradScaler('cuda', enabled=USE_MIXED_PRECISION)
    
    if rank == 0:
        logger.info(f"\nOptimization Configuration:")
        logger.info(f"  Batch size per GPU: {BATCH_SIZE}")
        logger.info(f"  Gradient accumulation steps: {ACCUMULATION_STEPS}")
        logger.info(f"  Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS * world_size}")
        logger.info(f"  Mixed precision training: {USE_MIXED_PRECISION}")
    
    # Create save directories
    if rank == 0:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    best_train_loss = float('inf')
    
    if rank == 0:
        logger.info(f"\nStarting training for {EPOCHS} epochs...")
        logger.info("=" * 70)
    
    for epoch in range(EPOCHS):
        # Set epoch to ensure different shuffling per epoch
        train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Show progress bar only for rank 0
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask_pos = batch["mask_positions"].to(device)
            
            # Mixed precision forward pass
            with autocast('cuda', enabled=USE_MIXED_PRECISION):
                logits = model(inputs)
                
                mask_logits = logits[mask_pos.bool()]
                mask_labels = labels[mask_pos.bool()]
                loss = criterion(mask_logits, mask_labels)
                
                # Gradient accumulation: loss needs normalization
                loss = loss / ACCUMULATION_STEPS
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Update parameters every ACCUMULATION_STEPS
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Statistics (restore real loss)
            train_loss += loss.item() * ACCUMULATION_STEPS
            
            # Calculate accuracy (no gradient needed, saves memory)
            with torch.no_grad():
                predictions = mask_logits.argmax(dim=-1)
                train_correct += (predictions == mask_labels).sum().item()
                train_total += mask_labels.size(0)
            
            if rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Aggregate training metrics across all GPUs
        train_loss_tensor = torch.tensor([avg_train_loss], device=device)
        train_acc_tensor = torch.tensor([train_accuracy], device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
        
        avg_train_loss = train_loss_tensor.item() / world_size
        train_accuracy = train_acc_tensor.item() / world_size
        
        # Learning rate scheduling
        scheduler.step()
        
        # Log metrics (all ranks log)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Print to console and save model only for rank 0
        if rank == 0:
            print("-" * 70)
            
            # Save best model (based on training loss)
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),  # Note: DDP model needs .module
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'vocab_size': vocab.vocab_size,
                }, MODEL_SAVE_PATH)
                logger.info(f"  ✓ Saved best model (train_loss: {avg_train_loss:.4f})")
            
            # Save checkpoint periodically
            if (epoch + 1) % 1 == 0:
                checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                }, checkpoint_path)
                logger.info(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # Ensure all processes synchronize
        dist.barrier()
    
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("Training Completed!")
        logger.info(f"Best Training Loss: {best_train_loss:.4f}")
        logger.info(f"Model saved to: {MODEL_SAVE_PATH}")
        logger.info("=" * 70)
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train Character Filling Language Model')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5',
                        help='GPU indices to use, comma-separated (e.g., 0,1,2,3,4,5)')
    parser.add_argument('--use_linformer', action='store_true',
                        help='Use Linformer attention (saves computation)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    args = parser.parse_args()
    
    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Get number of GPUs
    gpu_list = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_list)
    
    print(f"Using GPUs: {args.gpus}")
    print(f"Number of GPUs: {world_size}")
    print(f"Log directory: {args.log_dir}")
    print(f"Attention mechanism: {'Linformer' if args.use_linformer else 'Standard'}")
    
    # Launch multi-process training
    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()