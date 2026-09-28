"""
Training Pipeline for Neuromorphic Encryption Decryption Model
===============================================================
Trains the U-Net to learn the inverse mapping:
    autocorrelation (encrypted) → plaintext (decrypted)

Uses the physics-informed approach from the paper:
- Training data generated via simulated forward model
- NPCC loss function for optimization
- Multiple scattering conditions for generalization

Usage:
    python train.py --image_dir data/custom_images --epochs 50
    python train.py  # Uses auto-generated synthetic images
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.unet import UNet, NPCCLoss, CombinedLoss
from utils.dataset import NeuromorphicEncryptionDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (autocorr, gt) in enumerate(dataloader):
        autocorr = autocorr.to(device)
        gt = gt.to(device)
        
        optimizer.zero_grad()
        pred = model(autocorr)
        loss = criterion(pred, gt)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.6f}")
    
    return running_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for autocorr, gt in dataloader:
            autocorr = autocorr.to(device)
            gt = gt.to(device)
            pred = model(autocorr)
            loss = criterion(pred, gt)
            running_loss += loss.item()
            num_batches += 1
    
    return running_loss / max(num_batches, 1)


def save_sample_results(model, dataloader, device, save_dir, epoch):
    """Save visual comparison of encryption → decryption results."""
    model.eval()
    
    with torch.no_grad():
        autocorr, gt = next(iter(dataloader))
        autocorr = autocorr.to(device)
        pred = model(autocorr)
        
        # Take up to 4 samples
        n = min(4, autocorr.size(0))
        
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            # Autocorrelation (encrypted input)
            axes[i, 0].imshow(autocorr[i, 0].cpu().numpy(), cmap='hot')
            axes[i, 0].set_title('Encrypted\n(Autocorrelation)', fontsize=10)
            axes[i, 0].axis('off')
            
            # Decrypted output
            axes[i, 1].imshow(pred[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('Decrypted\n(U-Net Output)', fontsize=10)
            axes[i, 1].axis('off')
            
            # Ground truth
            axes[i, 2].imshow(gt[i, 0].numpy(), cmap='gray')
            axes[i, 2].set_title('Ground Truth\n(Original)', fontsize=10)
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Encryption → Decryption Results', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch+1}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Neuromorphic Encryption U-Net')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to custom images folder')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (default 128 for faster training; paper uses 256)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--samples_per_image', type=int, default=8,
                        help='Number of scattering conditions per image')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create dataset
    print("\n=== Creating Dataset ===")
    dataset = NeuromorphicEncryptionDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
        samples_per_image=args.samples_per_image,
        num_shifts_range=(2, 5),
        shift_pixels_range=(1, 4),
        augment=True
    )
    
    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    print("\n=== Creating U-Net Model ===")
    model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(npcc_weight=1.0, mse_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Step scheduler
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'image_size': args.image_size,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        
        # Save sample results every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_sample_results(model, val_loader, device, args.save_dir, epoch)
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1],
        'image_size': args.image_size,
    }, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    
    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss')
    ax.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.checkpoint_dir}/best_model.pth")
    print(f"Results saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
