import torch

X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

print(X.shape)

# -------------------------
# Virtual Node
# -------------------------
virtual_node = torch.zeros((1,3))

X_new = torch.cat([X, virtual_node], dim=0)

print(X_new)
print(X_new.shape)