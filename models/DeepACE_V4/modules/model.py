import torch
import torch.nn as nn
import torch.nn.functional as F
from netblocks import *
import math

class ChannelLayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x: (B, E, T)
        x = x.transpose(1, 2)  # now (B, T, E)
        x = self.ln(x)
        return x.transpose(1, 2)  # back to (B, E, T)

class CausalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # Padding is adjusted to ensure causality: only past information is used.
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
    def forward(self, x):
        # x shape: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class ConformerV2Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, conv_kernel_size, dropout=0.1):
        super().__init__()
        # First feed-forward module with scaling (macaron style)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # Multi-head self-attention (with causal mask applied in forward)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Convolution module; using ChannelLayerNorm to normalize over channel dimension
        self.conv = nn.Sequential(
            ChannelLayerNorm(embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU(),
            CausalConvolution(embed_dim, embed_dim, kernel_size=conv_kernel_size, dilation=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        
        # Second feed-forward module with scaling
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (T, B, E)
        # First feed-forward module (macaron-style, scaled by 0.5)
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention with causal mask.
        T, B, E = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = x + attn_out
        
        # Convolution module.
        # Rearrange x from (T, B, E) to (B, E, T) for Conv1d operations.
        x_conv = x.transpose(0, 1).transpose(1, 2)  # shape: (B, E, T)
        x_conv = self.conv(x_conv)
        # Convert back to (T, B, E)
        x_conv = x_conv.transpose(1, 2).transpose(0, 1)
        x = x + x_conv
        
        # Second feed-forward module (scaled by 0.5)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.base_max_len = max_len  # Base maximum length from your checkpoint
        
        # Precompute the positional encoding matrix for max_len positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Compute extra positions from self.pe.size(1) to seq_len
            extra_len = seq_len - self.pe.size(1)
            # Create a tensor for the additional positions
            position = torch.arange(self.pe.size(1), seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            # Use the same div_term formula
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) *
                                 (-math.log(10000.0) / self.d_model))
            extra_pe = torch.zeros(extra_len, self.d_model, device=x.device)
            extra_pe[:, 0::2] = torch.sin(position * div_term)
            extra_pe[:, 1::2] = torch.cos(position * div_term)
            extra_pe = extra_pe.unsqueeze(0)  # Shape: (1, extra_len, d_model)
            # Concatenate the original pe with the newly computed extra_pe
            pe = torch.cat([self.pe, extra_pe], dim=1)
        else:
            pe = self.pe
        
        # Add positional encoding to the input tensor (slice to the current sequence length)
        return x + pe[:, :seq_len]

class DeepACE(nn.Module):
    def __init__(
        self,
        L: int = 16,
        N: int = 512,
        P: int = 3,
        B: int = 128,
        S: int = 128,
        H: int = 512,
        X: int = 8,
        R: int = 3,
        M: int = 22,
        msk_activate: str = "sigmoid",
        causal: bool = True,
        num_conformer_blocks: int = 2,
        num_heads: int = 8,
        ff_hidden_dim: int = 2048,
        conv_kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.enc_num_feats = N
        self.enc_kernel_size = L
        self.enc_stride = L // 2
        self.out_channels = M

        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_feats,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.input_activation = Rectifier(self.enc_num_feats)

        # Positional encoding (applied to the sequence representation)
        self.pos_enc = PositionalEncoding(d_model=self.enc_num_feats)
        
        # Stack Conformer V2 blocks.
        self.conformer_v2_blocks = nn.ModuleList([
            ConformerV2Block(
                embed_dim=self.enc_num_feats,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_conformer_blocks)
        ])

        # ADDED: Conformer block before the mask generator.
        self.pre_mask_conformer = ConformerV2Block(
            embed_dim=self.enc_num_feats,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )

        self.mask_generator = MaskGenerator(
            input_dim=N,
            kernel_size=P,
            num_feats_bn=B,
            num_feats_skip=S,
            num_hidden=H,
            num_layers=X,
            num_stacks=R,
            msk_activate=msk_activate,
            causal=causal
        )

        # ADDED: Conformer block after the mask generator.
        self.post_mask_conformer = ConformerV2Block(
            embed_dim=self.enc_num_feats,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_feats,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        # ADDED: Channel balancing module (1D conv from out_channels to out_channels)
        self.channel_balancer = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1)
        self.out_activation = nn.Hardtanh(min_val=1e-6, max_val=1.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        padded, rest = self.pad_signal(input)
        feats = self.encoder(padded)
        feats = self.input_activation(feats)

        # Prepare for Conformer V2 blocks: (B, T, N)
        feats = feats.transpose(1, 2)
        feats = self.pos_enc(feats)
        # Convert to (T, B, N) for the Conformer blocks.
        feats = feats.transpose(0, 1)
        for block in self.conformer_v2_blocks:
            feats = block(feats)
        # Convert back to (B, N, T)
        feats = feats.transpose(0, 1).transpose(1, 2)

        # ADDED: Pre-mask Conformer block.
        # ConformerV2Block expects input shape (T, B, E), so transpose accordingly.
        feats_t = feats.transpose(1, 2).transpose(0, 1)  # (T, B, N)
        feats_pre = self.pre_mask_conformer(feats_t)
        feats_pre = feats_pre.transpose(0, 1).transpose(1, 2)  # (B, N, T)

        # Generate mask using the pre-processed features.
        mask = self.mask_generator(feats_pre)
        masked = feats_pre * mask

        # ADDED: Post-mask Conformer block.
        masked_t = masked.transpose(1, 2).transpose(0, 1)  # (T, B, N)
        masked_post = self.post_mask_conformer(masked_t)
        masked_post = masked_post.transpose(0, 1).transpose(1, 2)  # (B, N, T)

        # Decode, then apply channel balancing, then output activation.
        decoded = self.decoder(masked_post)
        balanced = self.channel_balancer(decoded)
        activated = self.out_activation(balanced)

        frames_left = self.enc_kernel_size // self.enc_stride
        def ceil_div(a, b):
            return -(-a // b)
        frames_right = ceil_div(rest + self.enc_stride, self.enc_stride)
        if frames_right > 0:
            output = activated[..., frames_left:-frames_right]
        else:
            output = activated[..., frames_left:]
        return output

    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.enc_kernel_size - (self.enc_stride + nsample % self.enc_kernel_size) % self.enc_kernel_size
        if rest > 0:
            pad = torch.zeros(batch_size, 1, rest, device=input.device, dtype=input.dtype)
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, 1, self.enc_stride, device=input.device, dtype=input.dtype)
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest
