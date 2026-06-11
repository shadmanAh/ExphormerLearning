import torch

from models.test.exphormer_layer import (
    ExphormerLayer
)

X = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8],
    [0.0, 0.0, 0.0]
])

mask = torch.tensor([
    [0,1,0,1,1],
    [1,0,1,0,1],
    [0,1,0,1,1],
    [1,0,1,0,1],
    [1,1,1,1,0]
])

layer = ExphormerLayer(
    in_dim=3,
    hidden_dim=8
)

output = layer(
    X,
    mask
)

print(output.shape)

print(output)