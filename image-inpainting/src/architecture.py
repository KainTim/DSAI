"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Convolutional block with Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """Downsampling block with two conv blocks and max pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        skip = self.conv2(self.conv1(x))
        return self.pool(skip), skip

class UpBlock(nn.Module):
    """Upsampling block with transposed conv and two conv blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)  # in_channels because of concatenation
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle dimension mismatch by interpolating x to match skip's size
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class MyModel(nn.Module):
    """U-Net style architecture for image inpainting"""
    def __init__(self, n_in_channels: int, base_channels: int = 64):
        super().__init__()
        
        # Initial convolution
        self.init_conv = ConvBlock(n_in_channels, base_channels)
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck1 = ConvBlock(base_channels * 8, base_channels * 16)
        self.bottleneck2 = ConvBlock(base_channels * 16, base_channels * 16)
        
        # Decoder (upsampling path)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        
        # Final upsampling and output
        self.final_up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.final_conv1 = ConvBlock(base_channels * 2, base_channels)
        self.final_conv2 = ConvBlock(base_channels, base_channels)
        
        # Output layer
        self.output = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # To ensure output is in [0, 1] range
    
    def forward(self, x):
        # Initial convolution
        x0 = self.init_conv(x)
        
        # Encoder
        x1, skip1 = self.down1(x0)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        # Bottleneck
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        
        # Decoder with skip connections
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Final layers
        x = self.final_up(x)
        # Handle dimension mismatch for final concatenation
        if x.shape[2:] != x0.shape[2:]:
            x = nn.functional.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        
        # Output
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x