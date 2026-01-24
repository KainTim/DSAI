"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """Initialize weights using Kaiming initialization for better training"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ChannelAttention(nn.Module):
    """Channel attention module (squeeze-and-excitation style)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class ResidualConvBlock(nn.Module):
    """Residual convolutional block for better gradient flow"""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class DownBlock(nn.Module):
    """Simplified downsampling block with conv blocks, residual connection, and max pooling"""
    def __init__(self, in_channels, out_channels, dropout=0.1, use_attention=True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
        self.residual = ResidualConvBlock(out_channels, dropout=dropout)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        skip = self.attention(x)
        return self.pool(skip), skip

class UpBlock(nn.Module):
    """Simplified upsampling block with transposed conv, residual connection, and conv blocks"""
    def __init__(self, in_channels, out_channels, dropout=0.1, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat: out_channels (from upconv) + in_channels (from skip)
        self.conv1 = ConvBlock(out_channels + in_channels, out_channels, dropout=dropout)
        self.residual = ResidualConvBlock(out_channels, dropout=dropout)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle dimension mismatch by interpolating x to match skip's size
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.residual(x)
        x = self.attention(x)
        return x

class MyModel(nn.Module):
    """Improved U-Net style architecture for image inpainting with attention and residual connections"""
    def __init__(self, n_in_channels: int, base_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # Initial convolution - simplified
        self.init_conv = nn.Sequential(
            ConvBlock(n_in_channels, base_channels, kernel_size=5, padding=2),
            ConvBlock(base_channels, base_channels)
        )
        
        # Encoder (downsampling path) - attention only on deeper layers
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout, use_attention=False)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout, use_attention=False)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout, use_attention=True)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16, dropout=dropout, use_attention=True)
        
        # Simplified bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 16, dropout=dropout),
            ConvBlock(base_channels * 16, base_channels * 16, dilation=2, padding=2, dropout=dropout),
            ResidualConvBlock(base_channels * 16, dropout=dropout),
            CBAM(base_channels * 16)
        )
        
        # Decoder (upsampling path) - attention only on deeper layers
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, dropout=dropout, use_attention=True)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, dropout=dropout, use_attention=True)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, dropout=dropout, use_attention=False)
        self.up4 = UpBlock(base_channels * 2, base_channels, dropout=dropout, use_attention=False)
        
        # Simplified final refinement layers
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        
        # Output layer with smooth transition
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        # Initial convolution
        x0 = self.init_conv(x)
        
        # Encoder
        x1, skip1 = self.down1(x0)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Handle dimension mismatch for final concatenation
        if x.shape[2:] != x0.shape[2:]:
            x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with initial features for better detail preservation
        x = torch.cat([x, x0], dim=1)
        x = self.final_conv(x)
        
        # Output
        x = self.output(x)
        
        return x