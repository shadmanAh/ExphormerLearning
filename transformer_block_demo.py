import torch
import torch.nn as nn

N = 5
hidden_dim = 8

H = torch.randn(N, hidden_dim)

attention_out = torch.randn(N, hidden_dim)

# Residual

H = H + attention_out

# LayerNorm

norm1 = nn.LayerNorm(hidden_dim)

H = norm1(H)

print(H)