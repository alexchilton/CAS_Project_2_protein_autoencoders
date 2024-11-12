import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np


class MolecularGVAE(nn.Module):

    def __init__(self, node_features=27, hidden_dim=64, latent_dim=32, max_nodes=500):
        super(MolecularGVAE, self).__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes

        # Encoder layers
        self.enc_conv1 = GATv2Conv(node_features, hidden_dim)
        self.enc_conv2 = GATv2Conv(hidden_dim, hidden_dim)

        # Node-level encoding
        self.node_mu = nn.Linear(hidden_dim, latent_dim)
        self.node_logvar = nn.Linear(hidden_dim, latent_dim)

        # Graph-level encoding
        self.graph_mu = nn.Linear(hidden_dim, latent_dim)
        self.graph_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers for node features with normalization
        self.dec_node_features = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_features)
        )

        # Edge prediction layers
        self.edge_pred = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        # Feature refinement network
        self.feature_refinement = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_features)
        )

    def encode(self, data):
        x, edge_index = data.x, data.edge_index

        # Node embeddings
        h = F.relu(self.enc_conv1(x, edge_index))
        h = F.relu(self.enc_conv2(h, edge_index))

        # Node-level latent variables
        node_mu = self.node_mu(h)
        node_logvar = self.node_logvar(h)

        # Graph-level pooling
        graph_h = torch.mean(h, dim=0)
        graph_mu = self.graph_mu(graph_h)
        graph_logvar = self.graph_logvar(graph_h)

        return node_mu, node_logvar, graph_mu, graph_logvar, h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z_nodes, num_nodes, node_embeddings=None):
        # Initial node features reconstruction
        node_features = self.dec_node_features(z_nodes)

        # Refine features if we have node embeddings (during training)
        if node_embeddings is not None:
            refinement_input = torch.cat([node_features, node_embeddings], dim=-1)
            node_features = self.feature_refinement(refinement_input)

        # Generate edges
        edge_logits = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_input = torch.cat([z_nodes[i], z_nodes[j]], dim=-1)
                edge_logit = self.edge_pred(edge_input)
                edge_logits.append(edge_logit)

        edge_logits = torch.cat(edge_logits, dim=0)
        return node_features, edge_logits

    def forward(self, data):
        # Encode
        node_mu, node_logvar, graph_mu, graph_logvar, node_embeddings = self.encode(data)

        # Sample latent variables
        z_nodes = self.reparameterize(node_mu, node_logvar)
        z_graph = self.reparameterize(graph_mu, graph_logvar)

        # Decode with node embeddings for feature refinement
        num_nodes = data.x.size(0)
        node_features, edge_logits = self.decode(z_nodes, num_nodes, node_embeddings)

        return node_features, edge_logits, node_mu, node_logvar, graph_mu, graph_logvar

    def generate(self, num_nodes, temperature=1.0, device='mps'):
        """
        Generate a new molecular graph with controlled randomness.

        Args:
            num_nodes: Number of nodes in the generated graph
            temperature: Controls the randomness of generation (higher = more random)
            device: Device to generate on
        """
        self.eval()  # Set to evaluation mode

        # Sample from latent space with temperature
        z_graph = torch.randn(self.latent_dim, device=device) * temperature
        z_nodes = torch.randn(num_nodes, self.latent_dim, device=device) * temperature

        # Decode
        node_features, edge_logits = self.decode(z_nodes, num_nodes)

        # Apply activation functions to ensure valid ranges
        # Assuming features need to be in [0, 1] range - adjust as needed
        node_features = torch.sigmoid(node_features)

        # Generate edges based on probabilities with temperature
        edge_probs = torch.sigmoid(edge_logits / temperature)
        edge_mask = torch.bernoulli(edge_probs)

        # Convert to sparse format
        edge_index = []
        edge_counter = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if edge_mask[edge_counter] > 0.5:
                    edge_index.extend([[i, j], [j, i]])
                edge_counter += 1

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, device=device).t()
        else:
            # Handle case with no edges
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)

        generated_data = Data(x=node_features, edge_index=edge_index)

        self.train()  # Set back to training mode
        return generated_data

    def loss_function(self, node_features, edge_logits, data, node_mu, node_logvar, graph_mu, graph_logvar):
        # Reconstruction loss for node features with feature-wise weighting
        feature_weights = torch.ones_like(data.x)  # You can adjust weights per feature if needed
        node_recon_loss = F.mse_loss(node_features * feature_weights,
                                     data.x * feature_weights,
                                     reduction='mean')

        # Convert edge_index to dense adjacency matrix for comparison
        true_adj = to_dense_adj(data.edge_index, max_num_nodes=node_features.size(0))[0]

        # Create predicted adjacency matrix
        pred_adj = torch.zeros_like(true_adj)
        edge_counter = 0
        num_nodes = node_features.size(0)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pred_prob = torch.sigmoid(edge_logits[edge_counter])
                pred_adj[i, j] = pred_prob
                pred_adj[j, i] = pred_prob
                edge_counter += 1

        # Edge prediction loss
        edge_recon_loss = F.binary_cross_entropy_with_logits(pred_adj[true_adj >= 0],
                                                             true_adj[true_adj >= 0])

        # KL divergence losses with annealing
        beta = 0.1  # Annealing factor - can be adjusted during training
        node_kl_loss = -0.5 * torch.mean(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        graph_kl_loss = -0.5 * torch.mean(1 + graph_logvar - graph_mu.pow(2) - graph_logvar.exp())

        # Total loss with weighted components
        total_loss = node_recon_loss + edge_recon_loss + beta * (node_kl_loss + graph_kl_loss)

        return total_loss, {
            'node_recon': node_recon_loss.item(),
            'edge_recon': edge_recon_loss.item(),
            'node_kl': node_kl_loss.item(),
            'graph_kl': graph_kl_loss.item()
        }
