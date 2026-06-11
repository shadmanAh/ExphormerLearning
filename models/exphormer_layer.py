import torch
import torch.nn as nn

from models.exphormer_attention import ExphormerAttention


class ExphormerLayer(nn.Module):

    def __init__(self, hidden_dim, ffn_multiplier=4):
        super().__init__()

        self.attention = ExphormerAttention(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim * ffn_multiplier, hidden_dim)
        )

    def forward(self, x, edge_index):
        # ------------------
        # Attention Block
        # ------------------
        attn_out = self.attention(x, edge_index)
        
        x = self.norm1(x + attn_out)

        # ------------------
        # FFN Block
        # ------------------
        ffn_out = self.ffn(x)

        x = self.norm2(x + ffn_out)

        return x