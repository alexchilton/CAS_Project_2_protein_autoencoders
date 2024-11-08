import networkx as nx
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_networkx
import torch
import random

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.data import DataLoader


def plot_graph(graph):
    # Convert PyTorch Geometric graph to NetworkX graph
    G = to_networkx(graph, to_undirected=True)

    # Use a circular layout
    pos = nx.circular_layout(G)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Generated Graph")
    plt.show()
class CircularGraphDataset:
    def __init__(self, node_sizes=[20, 30, 40, 50]):
        self.node_sizes = node_sizes
        self.graphs = self.create_graphs()

    def create_graphs(self):
        graphs = []
        for num_nodes in self.node_sizes:
            x = torch.rand(num_nodes, 2)  # Random features for each node (2 features)
            # Circular edge index: i -> (i+1) % num_nodes and reverse (i+1) % num_nodes -> i
            edge_index = torch.tensor(
                [[i, (i + 1) % num_nodes] for i in range(num_nodes)] +
                [[(i + 1) % num_nodes, i] for i in range(num_nodes)],
                dtype=torch.long
            ).t()  # Transpose to match (2, num_edges)
            graph = Data(x=x, edge_index=edge_index)
            graphs.append(graph)

            # Plot the graph
            #plot_graph(graph)
            # Print the size of the graphs array
        print(f"Number of graphs created: {len(graphs)}")
        return graphs


    def plot_graphs(self, graph):
        # Convert PyTorch Geometric graph to NetworkX graph
        G = to_networkx(graph, to_undirected=True)

        # Plot the graph
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Graph")
        plt.show()

    def get_graph(self, idx):
        return self.graphs[idx]

# Instantiate the dataset
dataset = CircularGraphDataset(node_sizes=[20, 30, 40, 50])





class GVAE(nn.Module):
    def __init__(self, hidden_dim=64):
        super(GVAE, self).__init__()
        self.hidden_dim = hidden_dim

        # Encoder: GCN layers
        self.encoder = GCNConv(2, hidden_dim)  # 2 input features for each node

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logstd = nn.Linear(hidden_dim, hidden_dim)

        # Decoder: MLP to reconstruct the graph
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Corrected input size
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)  # GCN
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        return mu, logstd

    def reparameterize(self, mu, logstd):
        std = torch.exp(0.5 * logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        # Reconstruct edge scores based on latent vector z
        z_i, z_j = z[edge_index[0]], z[edge_index[1]]  # Latent vectors of connected nodes
        z_concat = torch.cat([z_i, z_j], dim=-1)  # Concatenate latent vectors of two nodes

        # Decoder now accepts a tensor of shape (num_edges, 128)
        edge_score = torch.sigmoid(self.decoder(z_concat))  # Sigmoid for probability of edge
        return edge_score.squeeze()  # Flatten the result

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        edge_score = self.decode(z, edge_index)
        return edge_score, mu, logstd

    def loss_function(self, edge_score, edge_index, mu, logstd):
        # Negative log-likelihood loss for positive edges
        pos_edge_index = edge_index
        neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=edge_score.size(0), num_neg_samples=edge_index.size(1))

        # Compute positive loss (edge exists)
        pos_loss = -torch.log(edge_score[pos_edge_index[0]]).mean()  # Correct indexing for 1D tensor

        # Compute negative loss (edge does not exist)
        neg_loss = -torch.log(1 - edge_score[neg_edge_index[0]]).mean()  # Correct indexing for 1D tensor

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logstd - mu.pow(2) - logstd.exp())

        return pos_loss + neg_loss + kl_loss


# Initialize the GVAE model
model = GVAE(hidden_dim=2)

# Prepare data (circular graphs with varying sizes)
graph_dataset = CircularGraphDataset(node_sizes=[20, 30, 40, 50])
data_loader = DataLoader(graph_dataset.graphs, batch_size=1, shuffle=True)

optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for graph in data_loader:
        optimizer.zero_grad()

        edge_score, mu, logstd = model(graph)
        loss = model.loss_function(edge_score, graph.edge_index, mu, logstd)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")


def generate_random_graph(model, min_nodes=20, max_nodes=50):
    # Randomly choose a size for the graph
    num_nodes = random.randint(min_nodes, max_nodes)

    # Create a random graph with the chosen number of nodes (circular)
    x = torch.rand(num_nodes, 2)
    edge_index = torch.tensor(
        [[i, (i + 1) % num_nodes] for i in range(num_nodes)] +
        [[(i + 1) % num_nodes, i] for i in range(num_nodes)],
        dtype=torch.long
    ).t()

    # Generate the graph using the trained model
    model.eval()
    with torch.no_grad():
        graph = Data(x=x, edge_index=edge_index)
        edge_score, mu, logstd = model(graph)

        # Post-process and visualize if needed
        print(f"Generated graph with {num_nodes} nodes.")
        return edge_score, graph

# Example: Generate a random graph
#edge_score, graph = generate_random_graph(model, min_nodes=20, max_nodes=50)





# Example usage
edge_score, graph = generate_random_graph(model, min_nodes=20, max_nodes=50)
plot_graph(graph)



def visualize_latent_space(model, data):
    model.eval()
    with torch.no_grad():
        # Encode the graph to get the latent vectors
        mu, logstd = model.encode(data.x, data.edge_index)
        z = model.reparameterize(mu, logstd)

    # Use t-SNE to reduce the dimensionality to 2D
    num_samples = z.size(0)
    perplexity = min(30, num_samples - 1)  # Ensure perplexity is less than the number of samples
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    z_2d = tsne.fit_transform(z.cpu().numpy())

    # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c='skyblue', edgecolors='k')
    plt.title("Latent Space Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


visualize_latent_space(model, graph_dataset.get_graph(0))