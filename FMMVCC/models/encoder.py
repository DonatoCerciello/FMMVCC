import torch
from torch import nn
from mamba_ssm import Mamba


class MambaEncoder(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        n_layers=4,
        mask_mode=0,
        dropout=0.2,
        mode="unidirectional",
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.mask_mode = mask_mode
        self.dropout = dropout
        self.mode = mode

        # Embedding layer
        self.input_proj = nn.Linear(input_dims, hidden_dims)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dims, mode=self.mode)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dims)
        self.norm = nn.RMSNorm(hidden_dims) # Mamba paper uses RMSNorm
        self.dropout_layer = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        # x: [B, T, D]
        nan_mask = ~x.isnan().any(dim=-1)
        x = torch.nan_to_num(x, nan=0.0)

        # Input embedding
        h = self.input_proj(x)

        # Masking
        h = h * nan_mask.unsqueeze(-1)

        return self._forward_full(h)
    
    def _forward_full(self, h):
        """Processing full: output [B, T, D_out]"""
        for layer in self.layers:
            h = layer(h)

        h = self.dropout_layer(self.out_norm(h))
        h = self.out_proj(h)  # [B, T, D_out]
        return h


class MambaBlock(nn.Module):
    def __init__(self, dim, mode="unidirectional"):
        super().__init__()

        self.mode = mode
        self.norm = nn.RMSNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        if mode == "bidirectional":
            self.dir_gate = nn.Linear(dim, 1)

    def forward(self, x):

        x_norm = self.norm(x)

        out_fwd = self.mamba(x_norm)

        if self.mode == "bidirectional":
            out_bwd = torch.flip(
                self.mamba(torch.flip(x_norm, dims=[1])),
                dims=[1]
            )

            alpha = torch.sigmoid(self.dir_gate(x_norm))
            out = alpha * out_fwd + (1 - alpha) * out_bwd
        else:
            out = out_fwd

        return x + out