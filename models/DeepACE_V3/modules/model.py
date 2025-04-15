import torch
from netblocks import*

class DeepACE(torch.nn.Module):
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
        M: int = 22,
        msk_activate: str = "sigmoid",
        causal: bool = False,
        base_level = 0
    ):
        super().__init__()

        self.enc_num_feats = N
        self.enc_kernel_size = L
        self.enc_stride = L // 2
        self.out_channels = M
        self.base_level = base_level

        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels= self.enc_num_feats,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

        self.input_activation = Rectifier(self.enc_num_feats)

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
            in_channels=self.enc_num_feats,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.balance = ChannelRebalancer(self.out_channels)  
        self.out_activation = torch.nn.Hardtanh(min_val=1e-6, max_val=1.0) 
        self.se_block = SELayer(self.out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        # Pad the input and get the rest value
        padded, rest = self.pad_signal(input)
        feats = self.encoder(padded)
        feats = self.input_activation(feats)
        mask = self.mask_generator(feats)
        masked = feats * mask
        decoded = self.decoder(masked)
        output = self.balance(decoded)
        output = self.out_activation(output)
        
        #output = self.se_block(output)
        #output[output < self.base_level] = 0

        # Calculate the number of frames to remove from the start and end
        frames_left = self.enc_kernel_size // self.enc_stride
        # Function to perform ceiling division
        def ceil_div(a, b):
            return -(-a // b)

        frames_right = ceil_div(rest + self.enc_stride, self.enc_stride)

        # Remove the frames
        if frames_right > 0:
            output = output[..., frames_left:-frames_right]
        else:
            output = output[..., frames_left:]

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