from typing import Optional, Tuple

import torch

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
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = torch.arange(1, T + 1, device=input.device).view(1, -1) * C  # B, T
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum / entry_cnt) - cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

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
        causal: bool = False
    ):
        super().__init__()

        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        layers = [
            torch.nn.Conv1d(in_channels=bn_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
        ]

        if causal:
            layers.append(cLN(dimension=hidden_channels, eps=1e-8))
        else:
            layers.append(torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08))

        layers += [
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=0,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
        ]

        if causal:
            layers.append(cLN(dimension=hidden_channels, eps=1e-8))
        else:
            layers.append(torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08))

        self.conv_layers = torch.nn.Sequential(*layers)

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

        feature = self.conv_layers(input_padded)

        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)

        skip_out = self.skip_out(feature)
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
            bias = False
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


class ConvTasNet(torch.nn.Module):
    """
    Args:
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).
        causal (bool, optional): Causality contraint option.
    """

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
        msk_activate: str = "sigmoid",
        causal: bool = False
    ):
        super().__init__()
        self.enc_num_feats = N
        self.enc_kernel_size = L
        self.enc_stride = L // 2

        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels= self.enc_num_feats,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
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
            causal = causal
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels= self.enc_num_feats,
            out_channels=1,
            kernel_size=L,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )



    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        padded, num_pads = self.pad_signal(input) 
        feats = self.encoder(padded) 
        masked = self.mask_generator(feats) * feats
        output = self.decoder(masked) 
        

        start = self.enc_stride
        end = self.enc_stride + num_pads 

        if end > 0:
            output = output[...,start:-end] 
        else:
            output = output[...,start:]

        return output

    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size, _, nsample = input.size()
        stride = self.enc_stride
        kernel_size = self.enc_kernel_size

        # Calculate the total number of samples after padding
        total_samples = ((nsample - 1) // stride + 1) * stride
        rest = total_samples + kernel_size - stride - nsample
        if rest > 0:
            pad = torch.zeros(batch_size, 1, rest, device=input.device, dtype=input.dtype)
            input = torch.cat([input, pad], dim=2)

        # Pad additional samples at the beginning and end
        pad_aux = torch.zeros(batch_size, 1, stride, device=input.device, dtype=input.dtype)
        input = torch.cat([pad_aux, input, pad_aux], dim=2)

        return input, rest

