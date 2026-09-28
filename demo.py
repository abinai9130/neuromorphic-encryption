"""
Neuromorphic Encryption Demo
==============================
End-to-end demonstration:
  1. Encrypt an input image (simulated speckle + events + autocorrelation)
  2. Decrypt using trained U-Net
  3. Verify if the decrypted image matches the original
  4. Visualize the full pipeline

Usage:
    python demo.py --image path/to/image.png --model checkpoints/best_model.pth
    python demo.py  # Uses default test image
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.unet import UNet
from utils.speckle_physics import encrypt_image, generate_speckle_pattern


def load_image(path: str, size: int = 128) -> np.ndarray:
    """Load and preprocess an image."""
    img = Image.open(path).convert('L')
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.float64) / 255.0


def create_test_image(size: int = 128, content: str = 'N') -> np.ndarray:
    """Create a test image (letter on black background)."""
    from PIL import ImageDraw, ImageFont
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                  size=int(size * 0.7))
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), content, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), content, fill=255, font=font)
    return np.array(img).astype(np.float64) / 255.0


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a simplified SSIM."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def decrypt_image(model, autocorrelation: np.ndarray, device: torch.device) -> np.ndarray:
    """Decrypt an autocorrelation image using the trained U-Net."""
    model.eval()
    with torch.no_grad():
        # Prepare input
        inp = torch.from_numpy(autocorrelation.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        inp = inp.to(device)
        
        # Forward pass
        output = model(inp)
        
        # Convert back to numpy
        decrypted = output[0, 0].cpu().numpy()
    
    return decrypted


def run_full_pipeline(image: np.ndarray, model, device, key_seed: int = 42,
                      save_dir: str = 'outputs'):
    """Run the complete encrypt → decrypt → verify pipeline."""
    
    print("\n" + "=" * 60)
    print("NEUROMORPHIC ENCRYPTION PIPELINE")
    print("=" * 60)
    
    # === ENCRYPTION ===
    print("\n[1] ENCRYPTING IMAGE...")
    autocorr, intermediates = encrypt_image(
        image, key_seed=key_seed, num_shifts=3, shift_pixels=2
    )
    print(f"    Key seed: {key_seed}")
    print(f"    Speckle pattern shape: {intermediates['speckle'].shape}")
    print(f"    Accumulated events shape: {intermediates['accumulated_events'].shape}")
    print(f"    Autocorrelation shape: {autocorr.shape}")
    
    # === DECRYPTION ===
    print("\n[2] DECRYPTING WITH U-NET...")
    decrypted = decrypt_image(model, autocorr, device)
    print(f"    Decrypted image shape: {decrypted.shape}")
    
    # === VERIFICATION ===
    print("\n[3] VERIFYING DECRYPTION...")
    psnr = compute_psnr(image.astype(np.float32), decrypted)
    ssim = compute_ssim_simple(image.astype(np.float32), decrypted)
    correlation = np.corrcoef(image.flatten(), decrypted.flatten())[0, 1]
    
    print(f"    PSNR:  {psnr:.2f} dB")
    print(f"    SSIM:  {ssim:.4f}")
    print(f"    PCC:   {correlation:.4f}")
    
    # Threshold-based authentication
    auth_threshold = 0.5  # Correlation threshold for authentication
    is_authentic = correlation > auth_threshold
    print(f"\n    Authentication: {'PASS ✓' if is_authentic else 'FAIL ✗'} "
          f"(threshold: {auth_threshold}, correlation: {correlation:.4f})")
    
    # === WRONG KEY TEST ===
    print("\n[4] TESTING WITH WRONG KEY...")
    wrong_autocorr, _ = encrypt_image(image, key_seed=key_seed + 12345, 
                                       num_shifts=3, shift_pixels=2)
    wrong_decrypted = decrypt_image(model, wrong_autocorr, device)
    wrong_corr = np.corrcoef(image.flatten(), wrong_decrypted.flatten())[0, 1]
    wrong_auth = wrong_corr > auth_threshold
    print(f"    Wrong key correlation: {wrong_corr:.4f}")
    print(f"    Authentication: {'PASS ✓' if wrong_auth else 'FAIL ✗'} "
          f"(expected FAIL with wrong key)")
    
    # === VISUALIZATION ===
    print("\n[5] GENERATING VISUALIZATION...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: Full pipeline
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title('1. Original\n(Plaintext)', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    speckle_vis = np.log1p(intermediates['speckle'])
    ax2.imshow(speckle_vis, cmap='hot')
    ax2.set_title('2. Speckle Pattern\n(Coherent Scattering)', fontsize=11)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(intermediates['accumulated_events'], cmap='RdBu_r')
    ax3.set_title('3. Accumulated Events\n(Event Camera Sim)', fontsize=11)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(autocorr, cmap='hot')
    ax4.set_title('4. Autocorrelation\n(Encrypted Ciphertext)', fontsize=11)
    ax4.axis('off')
    
    # Row 2: Decryption results
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(autocorr, cmap='hot')
    ax5.set_title('5. Input to U-Net\n(Autocorrelation)', fontsize=11)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(decrypted, cmap='gray')
    ax6.set_title(f'6. Decrypted (Correct Key)\nPCC={correlation:.3f}', 
                  fontsize=11, fontweight='bold', color='green')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(wrong_decrypted, cmap='gray')
    ax7.set_title(f'7. Decrypted (Wrong Key)\nPCC={wrong_corr:.3f}', 
                  fontsize=11, color='red')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.imshow(image, cmap='gray')
    ax8.set_title('8. Ground Truth\n(Original)', fontsize=11)
    ax8.axis('off')
    
    plt.suptitle('Neuromorphic Encryption: Full Pipeline Demonstration\n'
                 '(Zhu et al., Advanced Photonics Nexus, 2024)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'full_pipeline_demo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved to: {save_path}")
    
    # === ROBUSTNESS TEST ===
    print("\n[6] ROBUSTNESS TEST (Noise & Cropping)...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Noise tests
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    for i, noise_var in enumerate(noise_levels):
        noisy_autocorr = autocorr + np.random.normal(0, noise_var, autocorr.shape)
        noisy_autocorr = np.clip(noisy_autocorr, 0, 1)
        noisy_dec = decrypt_image(model, noisy_autocorr, device)
        noisy_pcc = np.corrcoef(image.flatten(), noisy_dec.flatten())[0, 1]
        
        axes[0, i].imshow(noisy_dec, cmap='gray')
        axes[0, i].set_title(f'Noise σ²={noise_var}\nPCC={noisy_pcc:.3f}', fontsize=10)
        axes[0, i].axis('off')
    
    # Cropping tests
    crop_ratios = [1/16, 1/8, 1/4, 1/2]
    h, w = autocorr.shape
    for i, ratio in enumerate(crop_ratios):
        cropped = autocorr.copy()
        crop_h = int(h * ratio)
        crop_w = int(w * ratio)
        cropped[:crop_h, :crop_w] = 0  # Zero out a portion
        
        cropped_dec = decrypt_image(model, cropped, device)
        cropped_pcc = np.corrcoef(image.flatten(), cropped_dec.flatten())[0, 1]
        
        axes[1, i].imshow(cropped_dec, cmap='gray')
        axes[1, i].set_title(f'Crop {ratio:.2f}\nPCC={cropped_pcc:.3f}', fontsize=10)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Gaussian Noise', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Cropping', fontsize=12, fontweight='bold')
    
    plt.suptitle('Robustness Test: Decryption under Noise & Cropping', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'robustness_test.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved to: {save_path}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'correlation': correlation,
        'is_authentic': is_authentic,
        'wrong_key_correlation': wrong_corr,
    }


def main():
    parser = argparse.ArgumentParser(description='Neuromorphic Encryption Demo')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image to encrypt/decrypt')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (must match training)')
    parser.add_argument('--key', type=int, default=42,
                        help='Encryption key (random seed)')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--letter', type=str, default='N',
                        help='Letter for test image (if no --image given)')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    image_size = checkpoint.get('image_size', args.image_size)
    args.image_size = image_size  # Use the training size
    model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (trained for {checkpoint['epoch']+1} epochs, "
          f"val_loss: {checkpoint['val_loss']:.6f})")
    
    # Load or create test image
    if args.image:
        print(f"\nLoading image: {args.image}")
        image = load_image(args.image, size=image_size)
    else:
        print(f"\nCreating test image: letter '{args.letter}'")
        image = create_test_image(size=image_size, content=args.letter)
    
    # Run full pipeline
    results = run_full_pipeline(image, model, device, key_seed=args.key, 
                                save_dir=args.save_dir)
    
    return results


if __name__ == '__main__':
    main()
