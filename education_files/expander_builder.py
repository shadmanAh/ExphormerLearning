N = 8

for node in range(N):

    edge1 = (node + 2) % N
    edge2 = (node + 4) % N

    print(node, "->", edge1, edge2)