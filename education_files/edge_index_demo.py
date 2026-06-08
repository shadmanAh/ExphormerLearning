import torch

N = 8

edges = []

for node in range(N):

    edge1 = (node + 2) % N
    edge2 = (node + 4) % N

    edges.append((node, edge1))
    edges.append((node, edge2))

edge_index = torch.tensor(edges).t()

print(edge_index)

print("\nShape:")
print(edge_index.shape)