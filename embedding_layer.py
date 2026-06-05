import torch
import torch.nn as nn

# -------------------------
# Node Features
# -------------------------

X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

print("Input Shape:")
print(X.shape)

# -------------------------
# Embedding Layer
# -------------------------

embedding = nn.Linear(
    in_features=3,
    out_features=8
)

H = embedding(X)

print("\nOutput Shape:")
print(H.shape)

print("\nEmbedded Features:")
print(H)