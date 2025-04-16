from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch

class ChannelRebalancer(torch.nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.channel_mixer = torch.nn.Conv1d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_mixer(x)
        return x

class Rectifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(Rectifier, self).__init__()
        self.input_dim = input_dim

        self.kernel = torch.nn.Parameter(torch.empty(input_dim * 2, input_dim))

        torch.nn.init.kaiming_normal_(self.kernel, mode='fan_in', nonlinearity='relu')

        self.out_activation = torch.nn.Hardtanh(min_val=1e-6, max_val=1.0) 

    def forward(self, inputs):
        """
        Inputs shape: (batch_size, input_dim, length) or (batch_size, length, input_dim) depending on usage.
        """

        if inputs.shape[1] != self.input_dim:
            inputs = inputs.permute(0, 2, 1) 

        inputs = inputs - torch.mean(inputs, dim=-1, keepdim=True)

        pos = torch.nn.functional.relu(inputs)
        neg = torch.nn.functional.relu(-inputs)

        concatenated = torch.cat([pos, neg], dim=1) 
        abs_kernel = torch.abs(self.kernel) 

        concatenated = concatenated.permute(0, 2, 1) 
        mixed = torch.matmul(concatenated, abs_kernel) 
        mixed = mixed.permute(0, 2, 1)

        output = self.out_activation(mixed)
        
        return output


class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class cLN(torch.nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = torch.nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = torch.nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.register_buffer('gain', torch.ones(1, dimension, 1))
            self.register_buffer('bias', torch.zeros(1, dimension, 1))

    def forward(self, input):
        B, C, T = input.size()
        step_sum = input.sum(1)
        step_pow_sum = input.pow(2).sum(1)
        cum_sum = torch.cumsum(step_sum, dim=1) 
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)

        entry_cnt = torch.arange(1, T + 1, device=input.device).view(1, -1) * C
        cum_mean = cum_sum / entry_cnt 
        cum_var = (cum_pow_sum / entry_cnt) - cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean) / cum_std
        return x * self.gain + self.bias


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        bn_channels: int,
        skip_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int = 1,
        no_residual: bool = False,
        causal: bool = False,
        laurel_rank: int = 16  # Rank for low-rank augmentation
    ):
        super().__init__()

        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # First 1x1 convolution
        self.conv1x1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=bn_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            cLN(hidden_channels) if causal else torch.nn.GroupNorm(1, hidden_channels, eps=1e-8)
        )

        # Depthwise convolution
        self.depthwise_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=0,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            cLN(hidden_channels) if causal else torch.nn.GroupNorm(1, hidden_channels, eps=1e-8)
        )

        # Apply SE Layer
        self.se_layer = SELayer(hidden_channels)

        # LAuReL components
        # 1. Learnable scalar weights (LAuReL-RW)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

        # 2. Low-rank augmentation (LAuReL-LR)
        self.laurel_rank = laurel_rank
        self.A = torch.nn.Parameter(torch.randn(bn_channels, laurel_rank))
        self.B = torch.nn.Parameter(torch.randn(laurel_rank, bn_channels))

        # Residual and skip connections
        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(in_channels=hidden_channels, out_channels=bn_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=skip_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation
            input_padded = torch.nn.functional.pad(input, (padding, 0))
        else:
            padding_needed = (self.kernel_size - 1) * self.dilation
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            input_padded = torch.nn.functional.pad(input, (pad_left, pad_right))

        # Apply first convolution and depthwise convolution
        x = self.conv1x1(input_padded)
        x = self.depthwise_conv(x)

        # Apply SE Layer
        x = self.se_layer(x)

        # Residual connection with LAuReL augmentation
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(x)

            # LAuReL-RW: Apply learnable scalar weights
            residual = self.alpha * residual

            # LAuReL-LR: Apply low-rank augmentation to input
            input_reshaped = input.view(input.shape[0], input.shape[1], -1)
            augmentation = torch.matmul(self.B, input_reshaped)
            augmentation = torch.matmul(self.A, augmentation)
            augmentation = augmentation.view_as(input)

            # Combine with learnable beta
            residual = residual + self.beta * augmentation

        # Skip connection
        skip_out = self.skip_out(x)
        return residual, skip_out





class MaskGenerator(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        kernel_size: int,
        num_feats_bn: int,
        num_feats_skip: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
        causal: bool
    ):
        super().__init__()

        self.input_dim = input_dim

        self.num_feats_bn = num_feats_bn

        self.causal = causal

        if self.causal:
            self.input_norm = cLN(dimension=input_dim, eps=1e-8)
        else:
            self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)

        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats_bn, kernel_size=1)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.conv_layers.append(
                    ConvBlock(
                        bn_channels=num_feats_bn,
                        skip_channels=num_feats_skip,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                        causal=self.causal,
                    )
                )
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * dilation
        self.output_prelu = torch.nn.PReLU()
        
        self.adjust_channels_conv = torch.nn.Conv1d(
            in_channels=num_feats_skip,
            out_channels=num_feats_bn,
            kernel_size=1,
        )

        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats_bn,
            out_channels=input_dim,
            kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None: 
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        if output.shape[1] != self.num_feats_bn:
            output = self.adjust_channels_conv(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x expected shape: (T, B, embed_dim)
        T, B, E = x.shape
        # Create a causal mask: positions i can only attend to <= i.
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return attn_output
    

class CausalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # Padding is adjusted to ensure causality: only use past information.
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x shape: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class ChannelRebalancer(torch.nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.channel_mixer = torch.nn.Conv1d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_mixer(x)
        return x
    
class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, conv_kernel_size, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.self_attn = CausalSelfAttention(embed_dim, num_heads)
        self.conv_module = nn.Sequential(
            # Note: We assume the input is already in (B, channels, T) format.
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU(),
            CausalConvolution(embed_dim, embed_dim, kernel_size=conv_kernel_size, dilation=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape for self-attention: (T, B, E)
        residual = x
        # Feed-forward block 1
        x = x + 0.5 * self.ffn1(x)
        # Self-attention block (causal)
        x = x + self.self_attn(x)
        # Feed-forward block 2
        x = x + self.ffn2(x)
        
        # For convolution module, convert from (T, B, E) to (B, E, T)
        x_conv = x.transpose(0, 1)       # now (B, T, E)
        x_conv = self.layer_norm(x_conv)   # apply layer norm along last dim
        x_conv = x_conv.transpose(1, 2)     # now (B, E, T) for Conv1d
        
        x_conv = self.conv_module(x_conv)
        
        # Convert back to (T, B, E)
        x_conv = x_conv.transpose(1, 2).transpose(0, 1)
        
        return x + x_conv



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]
