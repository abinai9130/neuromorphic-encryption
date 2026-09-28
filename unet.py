"""
U-Net Architecture for Phase Retrieval (Decryption)
=====================================================
The paper uses U-Net as the learning-based phase retrieval model.
Input:  Autocorrelation of accumulated speckle events (256x256)
Output: Reconstructed plaintext image (256x256)

Loss:   Negative Pearson Correlation Coefficient (NPCC)

Reference: Zhu et al., APN 2024, Section 2.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive conv-BN-ReLU blocks (standard U-Net building block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net with skip connections for physics-informed phase retrieval.
    
    Architecture follows the standard encoder-decoder with skip connections.
    The paper notes: "Deep neural networks (DNN) with skip or residual
    connections excel at learning identity-like mappings" (Section 2.3).
    
    Args:
        in_channels: Input channels (1 for grayscale autocorrelation).
        out_channels: Output channels (1 for grayscale reconstructed image).
        features: Feature sizes at each encoder/decoder level.
    """
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.upconvs = nn.ModuleList()
        
        # Encoder path
        prev_ch = in_channels
        for f in features:
            self.encoder_blocks.append(DoubleConv(prev_ch, f))
            prev_ch = f
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder path
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder_blocks.append(DoubleConv(f * 2, f))
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder_blocks)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i](x)
        
        x = self.final_conv(x)
        x = torch.sigmoid(x)  # Output in [0, 1]
        return x


class NPCCLoss(nn.Module):
    """
    Negative Pearson Correlation Coefficient Loss.
    
    NPCC is used as the loss function per the paper (Section 2.3):
    "We select the negative Pearson correlation coefficient (NPCC) 
     as the loss function to optimize the DNN in the training process."
    
    NPCC = -Σ[(x - μ_x)(y - μ_y)] / [√(Σ(x - μ_x)²) · √(Σ(y - μ_y)²)]
    
    Minimizing NPCC maximizes the correlation between prediction and target.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Mean-center
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        # Pearson correlation
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = (
            torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8) *
            torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
        )
        
        pcc = numerator / denominator
        
        # Negative PCC (we want to minimize this, i.e., maximize correlation)
        npcc = -pcc.mean()
        
        return npcc


class CombinedLoss(nn.Module):
    """
    Combined loss: NPCC + MSE for stable training.
    NPCC handles structural similarity, MSE handles pixel accuracy.
    """
    def __init__(self, npcc_weight=1.0, mse_weight=0.5):
        super().__init__()
        self.npcc = NPCCLoss()
        self.mse = nn.MSELoss()
        self.npcc_weight = npcc_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        return self.npcc_weight * self.npcc(pred, target) + self.mse_weight * self.mse(pred, target)
