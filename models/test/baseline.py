import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class BaselineModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        x = self.encoder(x)

        graph_emb = global_mean_pool(x, batch)

        out = self.regressor(graph_emb)

        return out.squeeze(-1)