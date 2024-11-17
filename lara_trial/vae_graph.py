import torch
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv

class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_heads=4):
        """
        Graph Variational Autoencoder for mAb generation.

        Args:
            in_channels (int): Input feature size (node features).
            hidden_channels (int): Number of hidden units in GNN layers.
            latent_dim (int): Dimensionality of the latent space.
            num_heads (int): Number of attention heads in GATConv.
        """
        super(GraphVAE, self).__init__()
        # Encoder
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv_mu = GATConv(hidden_channels * num_heads, latent_dim, heads=1, concat=False)
        self.conv_logvar = GATConv(hidden_channels * num_heads, latent_dim, heads=1, concat=False)

        # Decoder
        self.decoder_fc = torch.nn.Linear(latent_dim, hidden_channels)
        self.node_reconstruct = torch.nn.Linear(hidden_channels, in_channels)  # Reconstruct node features
        self.edge_reconstruct = torch.nn.Linear(hidden_channels, 1)  # Reconstruct edge weights

    def encode(self, x, edge_index):
        # Encode graph into latent space
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        mu = self.conv_mu(x, edge_index)  # Mean of latent distribution
        logvar = self.conv_logvar(x, edge_index)  # Log variance of latent distribution
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        """
        Decode latent representation back into graphs.

        Args:
            z (torch.Tensor): Latent space representation (num_nodes x latent_dim).
            edge_index (torch.Tensor): Edge indices of the graph.

        Returns:
            torch.Tensor: Reconstructed node features.
            torch.Tensor: Reconstructed edge scores.
        """
        # Decode node features
        z = torch.relu(self.decoder_fc(z))  # Decode to hidden space
        node_features = self.node_reconstruct(z)  # Reconstruct node features

        # Decode edge features
        src, dest = edge_index  # Get source and destination nodes for edges
        edge_features = z[src] * z[dest]  # Combine latent features of connected nodes
        edge_scores = self.edge_reconstruct(edge_features).squeeze(-1)  # Reconstruct edge scores

        return node_features, edge_scores


    # def decode(self, z, batch_size):
    #     # Decode latent representation back into graphs
    #     z = torch.relu(self.decoder_fc(z))

    #     # Reconstruct nodes and edges
    #     node_features = self.node_reconstruct(z)
    #     edge_scores = self.edge_reconstruct(z).view(batch_size, -1)  # Dummy edge scores
    #     return node_features, edge_scores

    

    def forward(self, x, edge_index, batch):
        # Encoder
        mu, logvar = self.encode(x, edge_index)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        batch_size = batch.max().item() + 1
        node_features, edge_scores = self.decode(z, batch_size)

        return mu, logvar, z, node_features, edge_scores
