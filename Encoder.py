import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc_mu = torch.nn.Linear(hidden_channels, out_channels)
        self.fc_logvar = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar