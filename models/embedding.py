import torch
import torch.nn as nn


class AtomEncoder(nn.Module):

    def __init__(self, hidden_dim, num_embeddings):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_dim)

    def forward(self, x):
        x = x.squeeze(-1)
        return self.embedding(x)
    