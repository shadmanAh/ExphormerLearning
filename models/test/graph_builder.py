import torch


def add_global_node(edge_index, num_nodes):

    global_id = num_nodes

    src = []
    dst = []

    for node in range(num_nodes):

        src.append(global_id)
        dst.append(node)

        src.append(node)
        dst.append(global_id)

    global_edges = torch.tensor([src, dst], dtype=torch.long)

    edge_index = torch.cat([edge_index, global_edges], dim=1)

    return edge_index


def add_expander_edges(edge_index, num_nodes, k=3):

    src = []
    dst = []

    for i in range(num_nodes):

        neighbors = torch.randperm(num_nodes)[:k]

        for j in neighbors:

            if i != j:

                src.append(i)
                dst.append(j.item())

    expander_edges = torch.tensor([src, dst], dtype=torch.long)

    edge_index = torch.cat( [edge_index, expander_edges], dim=1)

    return edge_index


def build_exphormer_graph(edge_index, num_nodes,):

    edge_index = add_global_node(edge_index, num_nodes)

    edge_index = add_expander_edges(edge_index,num_nodes)

    return edge_index