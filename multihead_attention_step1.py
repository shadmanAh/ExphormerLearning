import torch
import torch.nn as nn
import math

N = 5

hidden_dim = 8
num_heads = 2
head_dim = 4

X = torch.randn(N, hidden_dim)

# Q K V

Wq = nn.Linear(hidden_dim, hidden_dim)
Wk = nn.Linear(hidden_dim, hidden_dim)
Wv = nn.Linear(hidden_dim, hidden_dim)

Q = Wq(X)
K = Wk(X)
V = Wv(X)
print("Q Bfore:")
print(Q)

# reshape

Q = Q.view(N, num_heads, head_dim)
K = K.view(N, num_heads, head_dim)
V = V.view(N, num_heads, head_dim)

print("Q shape:")
print(Q.shape)
print(Q)

# -------------------------
# Head 0
# -------------------------

Q0 = Q[:, 0, :]
K0 = K[:, 0, :]
V0 = V[:, 0, :]

print("\nQ0 shape:")
print(Q0.shape)

scores0 = (Q0 @ K0.T) / math.sqrt(head_dim)

print("\nHead0 Scores Shape:")
print(scores0.shape)