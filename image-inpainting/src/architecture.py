"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_weights(m):
    """Initialize weights using Kaiming initialization for better training"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class GatedSkipConnection(nn.Module):
    """Gated skip connection for better feature fusion"""
    def __init__(self, up_channels, skip_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(up_channels + skip_channels, up_channels, 1),
            nn.Sigmoid()
        )
        # Project skip to match up_channels if they differ
        if skip_channels != up_channels:
            self.skip_proj = nn.Conv2d(skip_channels, up_channels, 1)
        else:
            self.skip_proj = nn.Identity()
    
    def forward(self, x, skip):
        skip_proj = self.skip_proj(skip)
        combined = torch.cat([x, skip], dim=1)
        gate = self.gate(combined)
        return x * gate + skip_proj * (1 - gate)


class EfficientChannelAttention(nn.Module):
    """Efficient channel attention without dimensionality reduction"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global pooling
        y = self.avg_pool(x)
        # 1D convolution on channel dimension - add safety checks
        if y.size(-1) == 1 and y.size(-2) == 1:
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            y = torch.clamp(y, min=0.0, max=1.0)  # Ensure valid range
            return x * y.expand_as(x)
        return x


class SpatialAttention(nn.Module):
    """Efficient spatial attention module"""
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


class EfficientAttention(nn.Module):
    """Lightweight attention module combining channel and spatial"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = EfficientChannelAttention(channels)
        self.spatial_attn = SpatialAttention(kernel_size=5)
    
    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention module for long-range dependencies"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention map with numerical stability
        attention_logits = torch.bmm(query, key)
        # Scale for numerical stability
        attention_logits = attention_logits / math.sqrt(query.size(-1))
        attention = self.softmax(attention_logits)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, dropout=0.0, separable=False):
        super().__init__()
        if separable and in_channels > 1:
            # Depthwise separable convolution for efficiency
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        # Add momentum and eps for numerical stability
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5, track_running_stats=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class DenseBlock(nn.Module):
    """Lightweight dense block for better gradient flow"""
    def __init__(self, channels, growth_rate=8, num_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvBlock(channels + i * growth_rate, growth_rate, dropout=dropout))
        self.fusion = nn.Conv2d(channels + num_layers * growth_rate, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = self.fusion(torch.cat(features, dim=1))
        out = self.relu(self.bn(out))
        return out + x  # Residual connection

class ResidualConvBlock(nn.Module):
    """Improved residual convolutional block with pre-activation"""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class DownBlock(nn.Module):
    """Enhanced downsampling block with dense and residual connections"""
    def __init__(self, in_channels, out_channels, dropout=0.1, use_attention=True, use_dense=False, use_self_attention=False):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, dropout=dropout, separable=True)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
        if use_dense:
            self.dense = DenseBlock(out_channels, growth_rate=8, num_layers=2, dropout=dropout)
        else:
            self.dense = ResidualConvBlock(out_channels, dropout=dropout)
        self.attention = EfficientAttention(out_channels) if use_attention else nn.Identity()
        self.self_attention = SelfAttention(out_channels) if use_self_attention else nn.Identity()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense(x)
        x = self.attention(x)
        skip = self.self_attention(x)
        return self.pool(skip), skip

class UpBlock(nn.Module):
    """Enhanced upsampling block with gated skip connections"""
    def __init__(self, in_channels, out_channels, dropout=0.1, use_attention=True, use_dense=False, use_self_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Skip connection has in_channels, upsampled has out_channels
        self.gated_skip = GatedSkipConnection(out_channels, in_channels)
        # After gated skip: out_channels
        self.conv1 = ConvBlock(out_channels, out_channels, dropout=dropout, separable=True)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
        if use_dense:
            self.dense = DenseBlock(out_channels, growth_rate=8, num_layers=2, dropout=dropout)
        else:
            self.dense = ResidualConvBlock(out_channels, dropout=dropout)
        self.attention = EfficientAttention(out_channels) if use_attention else nn.Identity()
        self.self_attention = SelfAttention(out_channels) if use_self_attention else nn.Identity()
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle dimension mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.gated_skip(x, skip)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense(x)
        x = self.attention(x)
        x = self.self_attention(x)
        return x

class MyModel(nn.Module):
    """Enhanced U-Net architecture with dense connections and efficient attention"""
    def __init__(self, n_in_channels: int, base_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # Separate mask processing for better feature extraction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, base_channels // 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 4, base_channels // 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Image processing path
        self.image_conv = nn.Sequential(
            ConvBlock(3, base_channels, kernel_size=5, padding=2),
            ConvBlock(base_channels, base_channels)
        )
        
        # Fusion of mask and image features
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels + base_channels // 4, base_channels, 1),
            nn.BatchNorm2d(base_channels, momentum=0.1, eps=1e-5, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder with progressive feature extraction
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout, use_attention=False, use_dense=False)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout, use_attention=True, use_dense=True)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout, use_attention=True, use_dense=True, use_self_attention=True)
        
        # Enhanced bottleneck with multi-scale features, dense connections, and self-attention
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8, dropout=dropout),
            DenseBlock(base_channels * 8, growth_rate=12, num_layers=3, dropout=dropout),
            SelfAttention(base_channels * 8, reduction=4),
            ConvBlock(base_channels * 8, base_channels * 8, dilation=2, padding=2, dropout=dropout),
            ResidualConvBlock(base_channels * 8, dropout=dropout),
            EfficientAttention(base_channels * 8)
        )
        
        # Decoder with progressive reconstruction
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, dropout=dropout, use_attention=True, use_dense=True, use_self_attention=True)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, dropout=dropout, use_attention=True, use_dense=True)
        self.up3 = UpBlock(base_channels * 2, base_channels, dropout=dropout, use_attention=False, use_dense=False)
        
        # Multi-scale feature fusion with dense connections
        self.multiscale_fusion = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            DenseBlock(base_channels, growth_rate=8, num_layers=2, dropout=dropout//2),
            ConvBlock(base_channels, base_channels)
        )
        
        # Output with residual connection to input
        self.pre_output = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels // 2)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(base_channels // 2 + 3, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, 1),
            nn.Sigmoid()
        )
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        # Split input into image and mask
        image = x[:, :3, :, :]
        mask = x[:, 3:4, :, :]
        
        # Clamp inputs to valid range
        image = torch.clamp(image, 0.0, 1.0)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # Process mask and image separately
        mask_features = self.mask_conv(mask)
        image_features = self.image_conv(image)
        
        # Safety check after initial processing
        if not torch.isfinite(mask_features).all():
            mask_features = torch.nan_to_num(mask_features, nan=0.0, posinf=1.0, neginf=-1.0)
        if not torch.isfinite(image_features).all():
            image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Fuse features
        x0 = self.fusion(torch.cat([image_features, mask_features], dim=1))
        if not torch.isfinite(x0).all():
            x0 = torch.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Encoder
        x1, skip1 = self.down1(x0)
        if not torch.isfinite(x1).all():
            x1 = torch.nan_to_num(x1, nan=0.0, posinf=1.0, neginf=-1.0)
            skip1 = torch.nan_to_num(skip1, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x2, skip2 = self.down2(x1)
        if not torch.isfinite(x2).all():
            x2 = torch.nan_to_num(x2, nan=0.0, posinf=1.0, neginf=-1.0)
            skip2 = torch.nan_to_num(skip2, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x3, skip3 = self.down3(x2)
        if not torch.isfinite(x3).all():
            x3 = torch.nan_to_num(x3, nan=0.0, posinf=1.0, neginf=-1.0)
            skip3 = torch.nan_to_num(skip3, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Bottleneck
        x = self.bottleneck(x3)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Decoder with skip connections
        x = self.up1(x, skip3)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.up2(x, skip2)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.up3(x, skip1)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Handle dimension mismatch for final fusion
        if x.shape[2:] != x0.shape[2:]:
            x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        
        # Multi-scale fusion with initial features
        x = torch.cat([x, x0], dim=1)
        x = self.multiscale_fusion(x)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pre-output processing
        x = self.pre_output(x)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Concatenate with original masked image for residual learning
        x = torch.cat([x, image], dim=1)
        x = self.output(x)
        
        # Final safety clamp
        x = torch.clamp(x, 0.0, 1.0)
        
        return x