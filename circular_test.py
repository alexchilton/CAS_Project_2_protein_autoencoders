import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, negative_sampling
import networkx as nx
import matplotlib.pyplot as plt

class GraphDataset:
    def __init__(self, num_nodes=20, node_features=2):
        self.num_nodes = num_nodes
        self.node_features = node_features

    def create_data(self, num_nodes=20):
        # Create node features (2 features per node)
        x = torch.rand(num_nodes, 2)

        # Create circular edge connections: each node i is connected to i+1, and i-1
        edge_index = torch.tensor(
            [[i, (i + 1) % num_nodes] for i in range(num_nodes)] +
            [[(i + 1) % num_nodes, i] for i in range(num_nodes)],
            dtype=torch.long
        ).t()  # `.t()` transposes it to match the expected edge_index format

        # Convert to torch_geometric.data.Data format
        data = Data(x=x, edge_index=edge_index)

        # Split edges into train, validation, and test sets
        self.data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)

        # Generate negative samples for training
        num_neg_samples = self.data.train_pos_edge_index.size(1)  # Number of positive edges in training set
        train_neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples
        )

        # Store negative edge indices for training
        self.data.train_neg_edge_index = train_neg_edge_index

        # Check if there are any positive edges in the training set
        if self.data.train_pos_edge_index.size(1) == 0:
            raise ValueError("The training set has no edges. Try increasing the number of edges or adjusting the split ratios.")

        return self.data

# Instantiate the dataset and generate the graph data
graph_dataset = GraphDataset(num_nodes=20)
data = graph_dataset.create_data()

# Ensure the data is generated and check train_pos_edge_index
print("Train Pos Edge Index:")
print(data.train_pos_edge_index)

# Convert edge_index to a NetworkX graph for visualization
G = nx.Graph()
G.add_edges_from(data.train_pos_edge_index.t().tolist())

# Draw the circular graph
nx.draw_circular(G, with_labels=True, node_color="lightblue", edge_color="gray")
plt.show()
