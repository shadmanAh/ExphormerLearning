import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool

from models.embedding import AtomEncoder
from models.exphormer_layer import ExphormerLayer


class ExphormerModel(nn.Module):

    def __init__(self, hidden_dim=128, num_layers=3, num_embeddings = 12):
        super().__init__()

        self.encoder = AtomEncoder(hidden_dim, num_embeddings)

        self.layers = nn.ModuleList(
            [
                ExphormerLayer(hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        graph_embedding = global_mean_pool(x, batch)
        out = self.regressor(graph_embedding)
        return out.squeeze(-1)