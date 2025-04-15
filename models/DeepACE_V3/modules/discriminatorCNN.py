import torch
import torch.nn as nn
from utils import *

class ConvResidualBlock(nn.Module):
    """Improved residual block with dimension-preserving operations"""
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = ElectrodeNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = ElectrodeNorm(channels)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        return self.dropout(out + residual)

class ChannelAttention(nn.Module):
    """Channel attention focused on electrode relationships"""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        weights = self.attention(x.mean(-1))  # [B, C]
        return x * weights.unsqueeze(-1)  # [B, C, T]

class ElectrodeNorm(nn.Module):
    """Electrode-specific normalization"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, x):
        # x: [B, C, T]
        return self.norm(x)

class Discriminator(nn.Module):
    def __init__(self, 
                 input_size=22,
                 hidden_size=64,
                 num_layers=3,
                 dropout=0.2,
                 num_scales=3,
                 downsampling_factor=2):
        super().__init__()
        
        self.num_scales = num_scales
        # Initial projection (22 -> 128 channels)
        self.norm = ElectrodeNorm(input_size)
        self.attention = ChannelAttention(input_size)
        self.input_proj = nn.Conv1d(input_size, 2 * hidden_size, 1)
        

        # Create separate downsamplers & blocks per scale
        self.downsamplers = nn.ModuleList()
        self.blocks_per_scale = nn.ModuleList()
        for i in range(num_scales):
            # Downsampler
            downsampler = nn.Conv1d(
                    2 * hidden_size,
                    2 * hidden_size,
                    kernel_size=downsampling_factor ** i + 1,
                    stride=downsampling_factor ** i,
                    padding=(downsampling_factor ** i) // 2
                
            )
            self.downsamplers.append(downsampler)

            # Residual blocks for this scale
            blocks_for_this_scale = nn.ModuleList([
                ConvResidualBlock(2 * hidden_size, dropout)
                for _ in range(num_layers)
            ])
            self.blocks_per_scale.append(blocks_for_this_scale)

        # Output layers
        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, return_features=False):
        # Initial projection
        x = self.norm(x)
        x = self.attention(x)
        x = self.input_proj(x)  # [B, 128, T]
        

        all_logits = []
        all_features = []

        for i, downsample in enumerate(self.downsamplers):
            x_scaled = downsample(x)
            features = x_scaled

            # Pass through this scale's blocks
            for block in self.blocks_per_scale[i]:
                features = block(features)

            # Pool and classify
            pooled = self.attention_pool(features.permute(0, 2, 1))  # [B, C]
            logits = self.classifier(pooled)
            all_logits.append(logits)

            if return_features:
                all_features.append(features.permute(0, 2, 1))

        # Average across scales
        final_logits = torch.stack(all_logits, dim=1).mean(dim=1)
        
        if return_features:
            return final_logits, all_features
        else:
            return final_logits