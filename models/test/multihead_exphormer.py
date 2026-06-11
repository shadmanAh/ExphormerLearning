import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadExphormer(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_heads):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Embedding
        self.embedding = nn.Linear(in_dim, hidden_dim)

        # Q K V
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        # Output Projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, X, mask):

        # ----------------
        # Embedding
        # ----------------
        H = self.embedding(X)

        residual = H

        # ----------------
        # Q K V
        # ----------------
        Q = self.Wq(H)
        K = self.Wk(H)
        V = self.Wv(H)

        N = H.shape[0]

        Q = Q.view(N,self.num_heads,self.head_dim)
        K = K.view(N,self.num_heads,self.head_dim)
        V = V.view(N,self.num_heads,self.head_dim)

        outputs = []

        # ----------------
        # Multi Head
        # ----------------
        for head in range(self.num_heads):

            Qh = Q[:, head, :]
            Kh = K[:, head, :]
            Vh = V[:, head, :]

            scores = (Qh @ Kh.T) / math.sqrt(self.head_dim)

            scores = scores.masked_fill(mask == 0, -1e9)

            attention = F.softmax(scores, dim=1)

            out = attention @ Vh

            outputs.append(out)

        # ----------------
        # Concat
        # ----------------
        H = torch.cat(outputs,dim=1)

        # ----------------
        # Projection
        # ----------------
        H = self.out_proj(H)

        # ----------------
        # Residual + Norm
        # ----------------
        H = residual + H
        H = self.norm1(H)

        # ----------------
        # FFN
        # ----------------
        ffn_out = self.ffn(H)

        H = H + ffn_out
        H = self.norm2(H)

        return H