import torch
import torch.nn as nn
import torch.nn.functional as F
import math

N = 5

hidden_dim = 8
num_heads = 2
head_dim = 4

X = torch.randn(N, hidden_dim)

# -------------------
# Q K V
# -------------------

Wq = nn.Linear(hidden_dim, hidden_dim)
Wk = nn.Linear(hidden_dim, hidden_dim)
Wv = nn.Linear(hidden_dim, hidden_dim)

Q = Wq(X)
K = Wk(X)
V = Wv(X)

# -------------------
# reshape
# -------------------

Q = Q.view(N, num_heads, head_dim)
K = K.view(N, num_heads, head_dim)
V = V.view(N, num_heads, head_dim)

outputs = []

# -------------------
# each head
# -------------------

for head in range(num_heads):

    Qh = Q[:, head, :]
    Kh = K[:, head, :]
    Vh = V[:, head, :]

    scores = (Qh @ Kh.T) / math.sqrt(head_dim)

    attention = F.softmax(scores,dim=1)

    out = attention @ Vh

    outputs.append(out) # is a list

# -------------------
# concat heads
# -------------------

output = torch.cat(outputs, dim=1)

print("OutPuts: ")
print(outputs)
print("Final Output Shape:")
print(output.shape)
print("OutPut concated: ")
print(output)
