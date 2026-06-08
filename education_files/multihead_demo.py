import torch
import torch.nn as nn

N = 5

X = torch.randn(N, 8)

print("Input:")
print(X.shape)

num_heads = 2

head_dim = 4

Wq = nn.Linear(8, 8)

Q = Wq(X)

print("\nQ:")
print(Q.shape)

Q = Q.view(
    N,
    num_heads,
    head_dim
)

print("\nQ after reshape:")
print(Q.shape)