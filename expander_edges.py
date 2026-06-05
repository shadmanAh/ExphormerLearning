import torch
import random

num_nodes = 5

mask = torch.tensor([
    [0,1,0,1,1],
    [1,0,1,0,1],
    [0,1,0,1,1],
    [1,0,1,0,1],
    [1,1,1,1,0]
])

print("Original Mask")
print(mask)

# ------------------------
# Add random expander edges
# ------------------------

num_expander_edges = 2

for node in range(num_nodes):

    candidates = [
        i for i in range(num_nodes)
        if i != node
    ]

    chosen = random.sample(
        candidates,
        num_expander_edges
    )

    for dst in chosen:
        mask[node, dst] = 1

print("\nMask With Expander Edges")
print(mask)