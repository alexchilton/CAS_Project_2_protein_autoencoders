import torch
from torch_geometric.data import Data

def create_circular_graph(num_nodes=20):
    # Generate the edge indices for a circular graph
    edge_index = torch.tensor(
        [[i, (i + 1) % num_nodes] for i in range(num_nodes)] +
        [[(i + 1) % num_nodes, i] for i in range(num_nodes)],
        dtype=torch.long
    ).t()

    x = torch.rand(num_nodes, 2)  # Each node has 2 features
    data = Data(x=x, edge_index=edge_index)
    return data

# Use the function to create the circular graph
data = create_circular_graph(num_nodes=20)
print("Edge Index:")
print(data.edge_index)


import networkx as nx
import matplotlib.pyplot as plt

# Convert edge index to a NetworkX graph for visualization
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())

# Draw the circular graph
nx.draw_circular(G, with_labels=True, node_color="lightblue", edge_color="gray")
plt.show()