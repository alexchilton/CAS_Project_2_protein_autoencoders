import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


class GraphConstructor:
    def build_graph(self, residues, distance_threshold=None):
        """
        Build a PyTorch Geometric graph from residue data.

        Args:
            residues (list): List of residue dictionaries with features and coordinates.
            distance_threshold (float): Optional maximum distance for edges (e.g., 10.0 Å).
                                         If None, all residues are connected.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        # Extract node features and coordinates
        node_features = np.array([r["encoding"] for r in residues], dtype=np.float32)
        coordinates = np.array([r["coord"] for r in residues], dtype=np.float32)

        # Initialize edge lists
        edge_index = []
        edge_attr = []

        # Compute edges based on spatial proximity with a progress bar
        num_residues = len(coordinates)
        #print(f"Building graph for {num_residues} residues...")

        for i in tqdm(range(num_residues), desc="Computing edges"):
            for j in range(num_residues):
                if i != j:  # Exclude self-loops
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    if distance_threshold is None or dist <= distance_threshold:
                        edge_index.append([i, j])
                        edge_attr.append([1 / dist])  # Use inverse distance as edge weight

        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_features = torch.tensor(node_features, dtype=torch.float)

        # Create PyTorch Geometric Data object
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
