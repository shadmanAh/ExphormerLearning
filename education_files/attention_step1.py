import torch
import torch.nn as nn

# ------------------
# Node Features
# ------------------

X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

# ------------------
# Embedding
# ------------------

embedding = nn.Linear(3, 8)

H = embedding(X)

print("H shape:")
print(H.shape)

# ------------------
# Q K V
# ------------------

Wq = nn.Linear(8, 8)
Wk = nn.Linear(8, 8)
Wv = nn.Linear(8, 8)

Q = Wq(H)
K = Wk(H)
V = Wv(H)

print("\nQ shape:")
print(Q.shape)

print("\nK shape:")
print(K.shape)

print("\nV shape:")
print(V.shape)