import torch
import torch.nn as nn
import torch.nn.functional as F
import math

X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

embedding = nn.Linear(3, 8)

H = embedding(X)

Wq = nn.Linear(8, 8)
Wk = nn.Linear(8, 8)
Wv = nn.Linear(8, 8)

Q = Wq(H)
K = Wk(H)
V = Wv(H)

scores = (Q @ K.T) / math.sqrt(8)

attention = F.softmax(scores, dim=1)

output = attention @ V

print("Attention Shape:")
print(attention.shape)

print("\nOutput Shape:")
print(output.shape)

print("\nOutput:")
print(output)