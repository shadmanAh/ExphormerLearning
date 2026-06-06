import torch

from models.multihead_exphormer import MultiHeadExphormer

X = torch.tensor([
    [1.0,0.5,2.0],
    [0.3,1.5,0.7],
    [2.0,1.0,0.2],
    [1.2,0.1,1.8],
    [0.0,0.0,0.0]
])

mask = torch.tensor([
    [0,1,0,1,1],
    [1,0,1,0,1],
    [0,1,0,1,1],
    [1,0,1,0,1],
    [1,1,1,1,0]
])

model = MultiHeadExphormer(in_dim=3,hidden_dim=8,num_heads=2)

out = model.forward(X,mask)

print(out.shape)
print(out)