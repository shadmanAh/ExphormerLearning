import torch
import torch.nn as nn

from models.multihead_exphormer import MultiHeadExphormer

class ExphormerModel(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_heads, num_layers):

        super().__init__()

        self.layers = nn.ModuleList()

        # لایه اول
        self.layers.append(MultiHeadExphormer(in_dim=in_dim, hidden_dim=hidden_dim, num_heads=num_heads))

        # بقیه لایه‌ها
        for _ in range(num_layers - 1):

            self.layers.append(MultiHeadExphormer(in_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads))

    def forward(self, X, mask):
        H = X

        for layer in self.layers:
            H = layer(H, mask)

        return H