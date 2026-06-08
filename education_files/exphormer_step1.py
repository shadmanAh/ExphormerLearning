import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------
# 4 Nodes + 1 Virtual Node
# ---------------------------------

X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

virtual_node = torch.zeros((1, 3))

X = torch.cat([X, virtual_node], dim=0)

print("Input Shape:")
print(X.shape)

# ---------------------------------
# Embedding
# ---------------------------------

embedding = nn.Linear(3, 8)

H = embedding(X)

# ---------------------------------
# Q K V
# ---------------------------------

Q = nn.Linear(8, 8)(H)
K = nn.Linear(8, 8)(H)
V = nn.Linear(8, 8)(H)

# ---------------------------------
# Scores
# ---------------------------------

scores = (Q @ K.T) / math.sqrt(8)

# ---------------------------------
# Local + Virtual Node Mask
# ---------------------------------

mask = torch.tensor([
    [0,1,0,1,1],
    [1,0,1,0,1],
    [0,1,0,1,1],
    [1,0,1,0,1],
    [1,1,1,1,0]
])

scores = scores.masked_fill(mask == 0, -1e9)

attention = F.softmax(scores, dim=1)

output = attention @ V

print("\nAttention Shape:")
print(attention.shape)

print("\nOutput Shape:")
print(output.shape)

print("\nRow Sums:")
print(attention.sum(dim=1))