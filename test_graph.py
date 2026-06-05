import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 0)

print("Nodes:", G.nodes())
print("Edges:", G.edges())

nx.draw(G, with_labels=True)

plt.show()