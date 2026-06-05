import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# ------------------
# Q K V
# ------------------

Q = nn.Linear(8, 8)(H)
K = nn.Linear(8, 8)(H)
V = nn.Linear(8, 8)(H)

# ------------------
# Scores
# ------------------

scores = (Q @ K.T) / math.sqrt(8)

# ------------------
# Graph Mask
# ------------------

mask = torch.tensor([
    [0, 1, 0, 1],  # node 0
    [1, 0, 1, 0],  # node 1
    [0, 1, 0, 1],  # node 2
    [1, 0, 1, 0]   # node 3
])

# ------------------
# Apply Mask
# ------------------

scores = scores.masked_fill(mask == 0, -1e9)

attention = F.softmax(scores, dim=1)

output = attention @ V

print("Attention:")
print(attention)

print("\nRow sums:")
print(attention.sum(dim=1))

print("\nOutput shape:")
print(output.shape)