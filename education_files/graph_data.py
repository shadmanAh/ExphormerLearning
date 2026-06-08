import torch


class GraphData:

    def __init__(
        self,
        x,
        edge_index
    ):

        self.x = x

        self.edge_index = edge_index


x = torch.randn(8, 3)

edge_index = torch.tensor([
    [0,0,1,1],
    [2,4,3,5]
])

graph = GraphData(x, edge_index)

print(graph.x.shape)

print(graph.edge_index.shape)