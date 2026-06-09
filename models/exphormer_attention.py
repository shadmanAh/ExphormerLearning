import torch
import torch.nn as nn
import math


class ExphormerAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        src = edge_index[0]
        dst = edge_index[1]

        q = Q[src]
        k = K[dst]

        scores = (q * k).sum(dim=-1)

        scores = scores / math.sqrt(Q.shape[-1])

        return scores