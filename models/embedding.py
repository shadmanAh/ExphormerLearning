import torch
import torch.nn as nn


class AtomEncoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=13, embedding_dim=hidden_dim)

    def forward(self, x):

        x = x.squeeze(-1)

        return self.embedding(x)
    