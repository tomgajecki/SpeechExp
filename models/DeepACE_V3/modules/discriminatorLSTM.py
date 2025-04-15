import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils import AttentionPooling

class Discriminator(nn.Module):
    def __init__(self, 
                 input_size=22, 
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.2,
                 num_scales=1, 
                 downsampling_factor=2):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        self.downsampling_factor = downsampling_factor

        # Input projection with spectral norm for more stable training
        self.input_projection = spectral_norm(nn.Conv1d(
            in_channels=input_size,
            out_channels=2 * hidden_size,
            kernel_size=1
        ))

        self.input_group_norm = nn.GroupNorm(
            num_groups=2,
            num_channels=2 * hidden_size
        )

        self.input_dropout = nn.Dropout(dropout)

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Stacked BiLSTM layers with residual connections
        for _ in range(self.num_layers):
            lstm_layer = nn.LSTM(
                input_size=2 * hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.lstm_layers.append(lstm_layer)
            # Applying LayerNorm to stabilize outputs of each LSTM layer
            self.layer_norms.append(nn.LayerNorm(2 * hidden_size))
            self.dropouts.append(nn.Dropout(dropout))

        # Attention pooling for final representation per scale
        self.attention_pooler = AttentionPooling(2 * hidden_size)
        
        # Add a small dropout before attention pooling if desired:
        self.pre_attention_dropout = nn.Dropout(dropout)

        # Classifier (same across scales) with spectral norm
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(2 * hidden_size, hidden_size)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, 1))
        )

        self.scale_fusion_layer = nn.Sequential(
            nn.Linear(num_scales, num_scales),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(num_scales, 1)
        )

    def forward(self, x, return_features=False):
        # x: [batch, feature_dim, time_steps]
        scale_outputs = []
        scale_pooled_features = []

        for i in range(self.num_scales):
            scale = self.downsampling_factor ** i
            # Downsample in time dimension
            if scale > 1:
                scaled_x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            else:
                scaled_x = x
            # scaled_x: [batch, feature_dim, scaled_time_steps]

            # Project and normalize input
            projected = self.input_projection(scaled_x)  # [batch, 2*hidden_size, scaled_time_steps]
            projected = self.input_group_norm(projected)
            projected = self.input_dropout(projected)
            
            # Permute for LSTM
            projected = projected.permute(0, 2, 1) # [batch, scaled_time_steps, 2*hidden_size]

            # LSTM stack with residual connections
            residual_input = projected
            for lstm, ln, do in zip(self.lstm_layers, self.layer_norms, self.dropouts):
                out, _ = lstm(residual_input)
                out = ln(out)
                out = do(out)
                residual_input = residual_input + out

            final_rep = residual_input

            # Optional dropout before attention (uncomment if desired)
            final_rep = self.pre_attention_dropout(final_rep)

            # Attention pooling
            pooled = self.attention_pooler(final_rep)  # [batch, 2*hidden_size]

            # Classification at this scale
            out = self.fc(pooled)  # [batch, 1]
            out = out.squeeze(-1)  # [batch]

            scale_outputs.append(out)
            scale_pooled_features.append(pooled)

        # Combine scale-level outputs with a trainable fusion layer
        multi_scale_out = torch.stack(scale_outputs, dim=1)  # [batch, num_scales]
        final_out = self.scale_fusion_layer(multi_scale_out).squeeze(-1)  # [batch]

        if return_features:
            return final_out, scale_pooled_features
        else:
            return final_out
