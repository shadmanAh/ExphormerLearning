from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import math
from torch_geometric.nn import MessagePassing


class ExphormerAttention(MessagePassing):

    def __init__(self, hidden_dim):
        super().__init__(aggr="add")

        self.hidden_dim = hidden_dim

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        return self.propagate(
            edge_index,
            Q=Q,
            K=K,
            V=V
        )

    def message(self, Q_i, K_j, V_j, index):

        score = (Q_i * K_j).sum(dim=-1)
        score = score / math.sqrt(self.hidden_dim)

        # ✅ correct softmax usage
        alpha = softmax(score, index)

        return alpha.unsqueeze(-1) * V_j