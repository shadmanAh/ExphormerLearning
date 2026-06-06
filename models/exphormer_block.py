import torch
import torch.nn as nn


class ExphormerBlock(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, H, attention_out):
        # Residual 1
        H = H + attention_out

        # LayerNorm 1
        H = self.norm1(H)

        # FFN
        ffn_out = self.ffn(H)

        # Residual 2
        H = H + ffn_out

        # LayerNorm 2
        H = self.norm2(H)

        return H