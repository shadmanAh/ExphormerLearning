import torch
import torch.nn as nn

X = torch.randn(5,8)

print("Before:")
print(X)

norm = nn.LayerNorm(8)

Y = norm(X)

print("\nAfter:")
print(Y)