"""
Training V2 - More data, better generalization.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from models.unet import UNet, CombinedLoss
from utils.speckle_physics import encrypt_image


SIZE = 64
EPOCHS = 40
BS = 8
N_SCATTER = 8  # more scattering conditions


def gen_images(size=64):
    images = []
    chars = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
    for ch in chars:
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size*0.65))
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), ch, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((size-tw)//2, (size-th)//2 - bbox[1]), ch, fill=255, font=font)
        images.append(np.array(img, dtype=np.float32)/255.0)
    
    # Shapes
    for shape_type in ['circle', 'rect', 'triangle']:
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        m = size//5
        if shape_type == 'circle':
            draw.ellipse([m,m,size-m,size-m], fill=255)
        elif shape_type == 'rect':
            draw.rectangle([m,m,size-m,size-m], fill=255)
        else:
            draw.polygon([(size//2,m),(m,size-m),(size-m,size-m)], fill=255)
        images.append(np.array(img, dtype=np.float32)/255.0)
    
    return images


def gen_pair(image):
    ns = np.random.randint(2, 6)
    sp = np.random.randint(1, 4)
    autocorr, _ = encrypt_image(image, key_seed=np.random.randint(0,2**31), 
                                 num_shifts=ns, shift_pixels=sp)
    return autocorr.astype(np.float32), image.astype(np.float32)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check for custom images
    custom_dir = 'data/custom_images'
    if os.path.exists(custom_dir) and any(f.lower().endswith(('.jpg','.png','.jpeg','.bmp')) for f in os.listdir(custom_dir)):
        print("Loading custom images...")
        images = []
        for f in sorted(os.listdir(custom_dir)):
            if f.lower().endswith(('.jpg','.png','.jpeg','.bmp')):
                img = Image.open(os.path.join(custom_dir, f)).convert('L').resize((SIZE,SIZE))
                images.append(np.array(img, dtype=np.float32)/255.0)
        print(f"  Loaded {len(images)} custom images")
        if len(images) < 10:
            print("  Adding synthetic images for better training...")
            images.extend(gen_images(SIZE))
    else:
        print("Generating synthetic images...")
        images = gen_images(SIZE)
    
    print(f"Total base images: {len(images)}")
    
    # Pre-generate data with augmentation
    print(f"Generating {len(images)*N_SCATTER} training pairs...")
    t0 = time.time()
    all_x, all_y = [], []
    for img in images:
        for _ in range(N_SCATTER):
            aug = img.copy()
            if np.random.rand() > 0.5: aug = np.fliplr(aug).copy()
            k = np.random.randint(0, 4)
            aug = np.rot90(aug, k).copy()
            x, y = gen_pair(aug)
            all_x.append(x)
            all_y.append(y)
    
    X = torch.from_numpy(np.stack(all_x)[:,None,:,:])
    Y = torch.from_numpy(np.stack(all_y)[:,None,:,:])
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {X.shape}")
    
    # Split
    n = X.shape[0]
    perm = torch.randperm(n)
    sp = int(0.85*n)
    train_dl = DataLoader(TensorDataset(X[perm[:sp]], Y[perm[:sp]]), batch_size=BS, shuffle=True)
    val_dl = DataLoader(TensorDataset(X[perm[sp:]], Y[perm[sp:]]), batch_size=BS)
    print(f"Train: {sp}, Val: {n-sp}")
    
    model = UNet(1, 1, features=[32, 64, 128, 256]).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CombinedLoss(1.0, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-6)
    
    best_val = float('inf')
    tlosses, vlosses = [], []
    
    print(f"\nTraining {EPOCHS} epochs...")
    for ep in range(EPOCHS):
        model.train()
        tl = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_dl)
        tlosses.append(tl)
        
        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item()
        vl /= len(val_dl)
        vlosses.append(vl)
        sched.step()
        
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"Ep {ep+1:2d}/{EPOCHS} | Train: {tl:.5f} | Val: {vl:.5f}")
        
        if vl < best_val:
            best_val = vl
            torch.save({'epoch':ep, 'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'val_loss':vl, 'image_size':SIZE},
                       'checkpoints/best_model.pth')
    
    torch.save({'epoch':EPOCHS-1, 'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_loss':vlosses[-1], 'image_size':SIZE},
               'checkpoints/final_model.pth')
    
    # Visualize
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_dl))
        pred = model(xb.to(device))
        n = min(4, xb.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        if n==1: axes = axes.reshape(1,-1)
        for i in range(n):
            axes[i,0].imshow(xb[i,0].numpy(), cmap='hot')
            axes[i,0].set_title('Encrypted (Autocorrelation)')
            axes[i,0].axis('off')
            axes[i,1].imshow(pred[i,0].cpu().numpy(), cmap='gray')
            axes[i,1].set_title('Decrypted (U-Net)')
            axes[i,1].axis('off')
            axes[i,2].imshow(yb[i,0].numpy(), cmap='gray')
            axes[i,2].set_title('Ground Truth')
            axes[i,2].axis('off')
        plt.suptitle('Neuromorphic Encryption: Training Results', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig('outputs/training_results.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(tlosses)+1), tlosses, 'b-', label='Train')
    ax.plot(range(1,len(vlosses)+1), vlosses, 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (NPCC + MSE)')
    ax.set_title('Training Progress'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.close()
    
    print(f"\nComplete! Best val: {best_val:.5f}")
    print(f"Model: checkpoints/best_model.pth")


if __name__ == '__main__':
    main()
