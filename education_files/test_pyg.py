import torch
from torch_geometric.data import Data

x = torch.tensor([
    [1.0, 0.5, 2.0],
    [0.3, 1.5, 0.7],
    [2.0, 1.0, 0.2],
    [1.2, 0.1, 1.8]
])

edge_index = torch.tensor([
    [0,1,1,2,2,3,3,0],
    [1,0,2,1,3,2,0,3]
])

data = Data(x=x, edge_index=edge_index)

print(data)