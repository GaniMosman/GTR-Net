"""Base forecasters (Stage 1) used with GTR-Net: MLP, LSTM, TCN and Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_seq_len: int = 576,
        hidden_dim: int = 256,
        num_layers: int = 1,
        horizon: int = 288,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.input_size = input_size
        self.input_seq_len = input_seq_len
        self.horizon = horizon

        flat_in_dim = input_seq_len * input_size

        layers: list[nn.Module] = []
        curr_dim = flat_in_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            curr_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.level_head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.dim() != 3:
            raise ValueError(
                f"Expected x [batch, sequence, features], got {tuple(x.shape)}"
            )

        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"but received {x.size(-1)}."
            )

        if x.size(1) != self.input_seq_len:
            raise ValueError(
                f"Expected sequence length {self.input_seq_len}, "
                f"but received {x.size(1)}."
            )

        x_flat = x.reshape(x.size(0), -1)
        features = self.backbone(x_flat)
        y_hat = self.level_head(features)
        return y_hat


class LSTMForecaster(nn.Module):
    def __init__(
        self, input_size, hidden_dim=256, num_layers=2, horizon=288,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.level_head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, (hidden, _) = self.lstm(x)
        y_hat = self.level_head(self.dropout(hidden[-1]))
        return y_hat


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate=0.05):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.dropout1(self.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        out = self.dropout2(self.relu(self.conv2(F.pad(out, (self.pad, 0)))))
        return self.relu(out + self.residual(x))


class TCNForecaster(nn.Module):
    def __init__(
        self, input_channels, hidden_channels=128, kernel_size=3, num_blocks=8,
        horizon=288, dropout_rate=0.05,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.horizon = horizon
        self.tcn = nn.Sequential(*[
            ResidualTCNBlock(
                input_channels if index == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
                2 ** index,
                dropout_rate,
            )
            for index in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.level_head = nn.Linear(hidden_channels, horizon)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError(f"Expected x [batch, sequence, features], got {tuple(x.shape)}")
        if x.size(-1) != self.input_channels:
            raise ValueError(
                f"Model was created for {self.input_channels} input features, "
                f"but received {x.size(-1)}."
            )
        features = self.dropout(self.tcn(x.transpose(1, 2))[:, :, -1])
        y_hat = self.level_head(features)
        return y_hat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(
                position * divisor[:encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(
        self, input_size, input_seq_len=576, horizon=288, d_model=256,
        num_heads=4, num_layers=2, dim_feedforward=512, dropout_rate=0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.input_size = input_size
        self.input_seq_len = input_seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max(5000, input_seq_len))
        self.input_dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.level_head = nn.Linear(d_model, horizon)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, but received {x.size(-1)}."
            )
        encoded = self.input_projection(x) * math.sqrt(self.d_model)
        encoded = self.input_dropout(self.pos_encoder(encoded))
        features = self.dropout(self.final_norm(self.transformer_encoder(encoded)[:, -1, :]))
        y_hat = self.level_head(features)
        return y_hat


__all__ = [
    "MLPForecaster",
    "LSTMForecaster",
    "ResidualTCNBlock",
    "TCNForecaster",
    "PositionalEncoding",
    "TransformerForecaster",
]
