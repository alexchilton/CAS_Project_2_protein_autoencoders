import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
from ProteinAnalyzer import ProteinAnalyzer
class GraphCreator:
    def __init__(self, pdb_file, aa_info_file):
        self.pdb_file = pdb_file
        self.aa_info_file = aa_info_file

    def encode_aa(self, df):
        le = LabelEncoder()
        df['AA_encoded'] = le.fit_transform(df['AA'])
        return df, le

    def create_graph(self, df, distance_threshold=5.0):
        # Ensure all columns are numeric
        df[['X', 'Y', 'Z', 'AA_encoded','Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']] = df[['X', 'Y', 'Z', 'AA_encoded','Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']].apply(pd.to_numeric, errors='coerce')

        # Extract node features (XYZ coordinates, encoded AA, and other features)
        node_features = torch.tensor(df[['X', 'Y', 'Z', 'AA_encoded', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']].values, dtype=torch.float)

        # Calculate pairwise distances
        coordinates = df[['X', 'Y', 'Z']].values
        num_nodes = coordinates.shape[0]
        distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Define edges based on the distance threshold
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distance_matrix[i, j] <= distance_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create the Data object
        data = Data(x=node_features, edge_index=edge_index)
        return data

    def print_graph_metrics(self, graph):
        print(f"Number of nodes: {graph.num_nodes}")
        print(f"Number of edges: {graph.num_edges}")
        print(f"Node features shape: {graph.x.shape}")

    def draw_graph(self, graph, le):

        g  = nx.Graph()
        edge_index = graph.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            g.add_edge(edge_index[0, i], edge_index[1, i])

        pos = nx.spring_layout(g)
        node_colors = [plt.cm.tab20(graph.x[i, 3].item() % 20) for i in range(graph.num_nodes)]  # Color code nodes based on AA_encoded

        nx.draw(g, pos, with_labels=True, labels={i: le.inverse_transform([int(graph.x[i, 3].item())])[0] for i in range(graph.num_nodes)}, node_color=node_colors, node_size=50, font_size=8)
        plt.show()


pdb_file = '/Users/alexchilton/Downloads/archive/train/AF-D0ZA02-F1-model_v4.pdb'
aa_info_file = 'aa_mass_letter.csv'
analyzer = ProteinAnalyzer(pdb_file, aa_info_file)
graph_creator = GraphCreator(pdb_file, aa_info_file)

# Generate the autoencoder input DataFrame
autoencoder_input_df = analyzer.prepare_autoencoder_input()
# Example usage:
# graph_creator = GraphCreator(pdb_file, aa_info_file)
autoencoder_input_df, le = graph_creator.encode_aa(autoencoder_input_df)
pd.set_option('display.max_columns', None)

print(autoencoder_input_df.shape)
print(autoencoder_input_df.head())

# Unpack the returned tuple
graph = graph_creator.create_graph(autoencoder_input_df)

# Print graph metrics
graph_creator.print_graph_metrics(graph)

# Draw the graph
graph_creator.draw_graph(graph, le)