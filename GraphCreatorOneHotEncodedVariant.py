import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
from ProteinAnalyzer import ProteinAnalyzer
class GraphCreatorOneHotEncodedVariant:

    def create_graph(self, df, distance_threshold=5.0):
        # Ensure all columns are numeric
        df[['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count', 'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G', 'AA_H', 'AA_I', 'AA_K', 'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q', 'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y']]\
            = df[['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count', 'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G', 'AA_H', 'AA_I', 'AA_K', 'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q', 'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y']].apply(pd.to_numeric, errors='coerce')

        # Extract node features (XYZ coordinates, encoded AA, and other features)
        node_features = torch.tensor(df[['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count', 'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G', 'AA_H', 'AA_I', 'AA_K', 'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q', 'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y']].values, dtype=torch.float)

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

    def draw_graph(self, graph, onehot_encoder):
        g = nx.Graph()
        edge_index = graph.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            g.add_edge(edge_index[0, i], edge_index[1, i])

        pos = nx.spring_layout(g)

        # Adjust the column names to match the number of features in graph.x
        num_features = graph.x.size(1)
        columns = ['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count', 'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G', 'AA_H', 'AA_I', 'AA_K', 'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q', 'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y']

        # Decode the one-hot encoded values
        encoded_df = pd.DataFrame(graph.x.numpy(), columns=columns)

        # Verify the column names in the DataFrame
        #print("Columns in the DataFrame:")
        #print(encoded_df.columns)

        # Ensure the OneHotEncoder is correctly applied

        #encoded_columns = onehot_encoder.get_feature_names_out(['AA'])

        #print("Expected one-hot encoded columns:")
        #print(encoded_columns)

        # Check if the expected columns are in the DataFrame
        #missing_columns = [col for col in encoded_columns if col not in encoded_df.columns]
        #if missing_columns:
        #    print("Missing columns in the DataFrame:")
        #    print(missing_columns)
        #else:
        #    print("All expected columns are present in the DataFrame.")

        decoded_df = self.decode_values(encoded_df, onehot_encoder, original_column_name='AA')

        # Use the decoded values for node names and colors
        node_labels = {i: decoded_df['AA'][i] for i in range(graph.num_nodes)}
        unique_labels = decoded_df['AA'].unique()
        color_map = {label: plt.cm.tab20(i % 20) for i, label in enumerate(unique_labels)}
        node_colors = [color_map[decoded_df['AA'][i]] for i in range(graph.num_nodes)]

        nx.draw(g, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=200, font_size=8)
        plt.show()

    # Update the decode_values function
    def decode_values(self, encoded_df, onehot_encoder, original_column_name='AA'):
        encoded_columns = onehot_encoder.get_feature_names_out([original_column_name])
        encoded_values = encoded_df[encoded_columns].values
        decoded_values = onehot_encoder.inverse_transform(encoded_values)
        decoded_df = encoded_df.drop(columns=encoded_columns)
        decoded_df[original_column_name] = decoded_values.flatten()
        return decoded_df

    def pad_graphs(self,  graphs, max_nodes):
        padded_graphs = []
        for graph in graphs:
            num_nodes = graph.x.size(0)
            if num_nodes < max_nodes:
                # Pad node features
                padding = torch.zeros((max_nodes - num_nodes, graph.x.size(1)))
                padded_x = torch.cat([graph.x, padding], dim=0)

                # Pad edge indices
                padded_edge_index = graph.edge_index

                # Create padded graph
                padded_graph = Data(x=padded_x, edge_index=padded_edge_index)
                padded_graphs.append(padded_graph)
            else:
                padded_graphs.append(graph)
        return padded_graphs
