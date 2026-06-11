import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExphormerLayer(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_heads):

        super().__init__()
        #تعداد سر 
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Node Embedding
        self.embedding = nn.Linear(in_dim, hidden_dim)

        # Q K V
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, mask):

        H = self.embedding(X)

        Q = self.Wq(H)
        K = self.Wk(H)
        V = self.Wv(H)

        scores = ( Q @ K.T) / math.sqrt(Q.shape[1])

        scores = scores.masked_fill( mask == 0, -1e9)

        attention = F.softmax(scores, dim=1)

        output = attention @ V

        return output