import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, z, edge_index):
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x