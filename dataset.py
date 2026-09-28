"""
Dataset & Data Generation
==========================
Generates synthetic training data using the forward physics model.

The paper notes (Section 2.4):
"In order to reduce the cost of data collection, a synthetic framework
is proposed to generate the simulated data for DNN training."

"A total of nine groups of speckle events are generated, which have 
distinct degraded processes with different scattering transport modes."
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from utils.speckle_physics import generate_training_pair


class NeuromorphicEncryptionDataset(Dataset):
    """
    On-the-fly synthetic dataset for training the decryption U-Net.
    
    For each sample, takes a ground truth image, applies the forward
    encryption model with random scattering conditions, and returns
    the (autocorrelation, ground_truth) pair.
    
    Supports:
    - Custom images from a folder
    - Auto-generated letter/shape images (if no custom images provided)
    - Multiple scattering conditions per image (data augmentation)
    """
    
    def __init__(self, image_dir: str = None, image_size: int = 256, 
                 samples_per_image: int = 8, num_shifts_range=(2, 5),
                 shift_pixels_range=(1, 4), augment: bool = True):
        """
        Args:
            image_dir: Path to folder with custom images (jpg/png).
                       If None, generates letter images automatically.
            image_size: Target image size (default 256x256 per the paper).
            samples_per_image: Number of different scattering conditions per image.
            num_shifts_range: Range for number of motion shifts.
            shift_pixels_range: Range for shift amount in pixels.
            augment: Whether to apply random augmentations.
        """
        self.image_size = image_size
        self.samples_per_image = samples_per_image
        self.num_shifts_range = num_shifts_range
        self.shift_pixels_range = shift_pixels_range
        self.augment = augment
        
        self.images = []
        
        if image_dir and os.path.exists(image_dir):
            # Load custom images
            valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for fname in sorted(os.listdir(image_dir)):
                if os.path.splitext(fname)[1].lower() in valid_ext:
                    path = os.path.join(image_dir, fname)
                    img = Image.open(path).convert('L')  # Grayscale
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                    arr = np.array(img).astype(np.float32) / 255.0
                    self.images.append(arr)
            
            if len(self.images) == 0:
                print(f"No images found in {image_dir}, generating synthetic images...")
                self.images = self._generate_synthetic_images()
        else:
            print("No image directory provided, generating synthetic images...")
            self.images = self._generate_synthetic_images()
        
        print(f"Dataset: {len(self.images)} base images × {samples_per_image} scattering conditions = {len(self)} samples")
    
    def _generate_synthetic_images(self, count=50):
        """Generate diverse synthetic images for training."""
        images = []
        size = self.image_size
        
        # Letters (similar to the paper's N, L, H, 3, 5, X targets)
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for letter in letters:
            img = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(img)
            # Use default font, scale text to fill image
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(size * 0.7))
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), letter, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (size - tw) // 2
            y = (size - th) // 2 - bbox[1]
            draw.text((x, y), letter, fill=255, font=font)
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(arr)
        
        # Simple shapes
        shapes = ['circle', 'square', 'triangle', 'star', 'cross']
        for shape in shapes:
            img = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(img)
            margin = size // 6
            
            if shape == 'circle':
                draw.ellipse([margin, margin, size - margin, size - margin], fill=255)
            elif shape == 'square':
                draw.rectangle([margin, margin, size - margin, size - margin], fill=255)
            elif shape == 'triangle':
                points = [(size // 2, margin), (margin, size - margin), (size - margin, size - margin)]
                draw.polygon(points, fill=255)
            elif shape == 'star':
                cx, cy, r = size // 2, size // 2, size // 3
                points = []
                for i in range(10):
                    angle = np.pi / 2 + i * np.pi / 5
                    rad = r if i % 2 == 0 else r // 2
                    points.append((int(cx + rad * np.cos(angle)), int(cy - rad * np.sin(angle))))
                draw.polygon(points, fill=255)
            elif shape == 'cross':
                t = size // 6
                draw.rectangle([size // 2 - t, margin, size // 2 + t, size - margin], fill=255)
                draw.rectangle([margin, size // 2 - t, size - margin, size // 2 + t], fill=255)
            
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(arr)
        
        # Random noise patterns
        for _ in range(9):
            arr = np.random.rand(size, size).astype(np.float32)
            arr = (arr > 0.7).astype(np.float32)  # Binary random pattern
            images.append(arr)
        
        return images
    
    def __len__(self):
        return len(self.images) * self.samples_per_image
    
    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        scattering_idx = idx % self.samples_per_image
        
        image = self.images[img_idx].copy()
        
        # Apply augmentation if enabled
        if self.augment:
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
        
        # Random scattering parameters (different conditions per paper)
        num_shifts = np.random.randint(*self.num_shifts_range)
        shift_pixels = np.random.randint(*self.shift_pixels_range)
        
        # Generate encrypted autocorrelation
        autocorr, ground_truth = generate_training_pair(
            image,
            key_seed=None,  # Random key each time
            num_shifts=num_shifts,
            shift_pixels=shift_pixels,
            target_size=self.image_size
        )
        
        # Convert to tensors (add channel dimension)
        autocorr_tensor = torch.from_numpy(autocorr).unsqueeze(0)  # (1, H, W)
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0)     # (1, H, W)
        
        return autocorr_tensor, gt_tensor
