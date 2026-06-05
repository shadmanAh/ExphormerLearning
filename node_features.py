import torch

# چهار نود
X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

print("Node Features:")
print(X)

print()
print("Shape:")
print(X.shape)