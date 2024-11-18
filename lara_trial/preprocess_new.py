import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.PDBParser import PDBParser
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)


import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.PDBParser import PDBParser
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

class ProteinAnalyzer3:
    def __init__(self, pdb_file, aa_info_file):
        """Initialize with additional data structures for better molecule organization"""
        self.pdb_file = pdb_file
        self.aa_info_file = aa_info_file
        self.parser = PDBParser(PERMISSIVE=1)
        self.structure = self.parser.get_structure('protein_structure', self.pdb_file)
        self.aa_info_dict = self.get_aa_info_dict()
        self.aa_info_dict_short = self.get_aa_info_dict_short()

        # Extract amino acid information first
        self.c_alpha_df = self.extract_c_alpha_info()

        # Create a dictionary to store molecules associated with each amino acid
        self.residue_molecules = self.create_residue_molecule_mapping()

        # Create molecule DataFrame with residue indices
        self.molecule_df = self.create_molecule_dataframe()

        # Add molecule counts to c_alpha_df
        self.add_molecule_counts_to_residues()

    def get_aa_info_dict(self):
        """Read the amino acid info file and return a dictionary with the amino acid properties."""
        aa_info = pd.read_csv(self.aa_info_file)
        return aa_info.set_index('Abbrev.').to_dict(orient='index')

    def get_aa_info_dict_short(self):
        """Read the amino acid info file and return a dictionary with short names."""
        aa_info = pd.read_csv(self.aa_info_file)
        return aa_info.set_index('Short').to_dict(orient='index')

    def get_residue_info(self, res_name):
        """Return the short name of the amino acid given the residue name."""
        if res_name in self.aa_info_dict:
            return self.aa_info_dict[res_name]['Short']
        return None

    def process_residue(self, residue, res_name):
        """Process a residue and extract the C-alpha atom coordinates."""
        if res_name not in self.aa_info_dict:
            return None

        if 'CA' in residue:
            ca_atom = residue['CA']
            x, y, z = ca_atom.get_coord()
            return [
                x, y, z,
                self.aa_info_dict[res_name]['Short'],
                self.aa_info_dict[res_name]['Avg. mass (Da)']
            ]
        return None

    def extract_c_alpha_info(self):
        """Extract the C-alpha atom coordinates and amino acid properties from the PDB file."""
        small_molecules = {
            "HOH", "P4G", "GOL", "EDO", "UDP", "GLC", "MN", "MG", "FOL", "NAG", "SO4", "ZN",
            "NAG", "CA", "DKA", "AR7", "0QE", "BMA", "D67", "CLR", "PO4","AGH",
            "POG", "1WV", "KBY", "ATP", "HIC", "UNK", "FNC", "IOD", "CNC", "CL",
            "EPE", "FUC", "RET", "MAN", "NA", "K", "CR2", "CRO", "CIT", "PEG", "ACT",
            "PDO", "PIO", "ABU", "DZP", "1IO", "LDP", "GTP", "IPA", "UPG", "XQ2",
            "CDL", "BCG", "LMT", "OCT", "1PE", "PG4", "D12", "D10", "HEX",
            "1IO", "BGC", "PLM", "NH2", "Y01", "DMS", "G3C", "LVW", "W20", "HG", "TYS", "ACE", "SEP",
            "TPO", "9EG", "43I", "IAC", "SCN", "PLM", "MYR", "PLC", "FAD", "HEM",
            "LBN", "CAC", "IMD", "GSP", "Y00", "MGO", "P2E", "IP9", "RV2", "ACY", "2CV",
            "FMT", "FVK", "NRQ", "UBL", "CU", "RR6", "ADP", "AF3", "GAL", "GLA", "5FW",
            "CFF", "TRS", "HCY", "AXL", "OLC", "UNL", "SIN", "1I8", "WZ0", "NG0", "PGE",
            "D6M", "PTY", "P42", "ACP", "LNR", "N7P", "6EA", "9PG", "BNG", "RTO", "TWT",
            "TRD", "MSE", "P0G", "FLC", "RTV", "LDP", "HSM", "PCA", "MTX", "B3P", "Y9Q",
            "CVV", "OLA", "6EA", "LPD"
        }

        c_alpha_matrix = []
        res_list = Selection.unfold_entities(self.structure, "R")

        for residue in res_list:
            if residue.get_resname() not in small_molecules:
                res_name = residue.get_resname().title()
                residue_data = self.process_residue(residue, res_name)
                if residue_data:
                    c_alpha_matrix.append(residue_data)

        return pd.DataFrame(
            c_alpha_matrix,
            columns=["X", "Y", "Z", "AA", "Mass"]
        )

    def create_residue_molecule_mapping(self):
        """Create a mapping of residues to their nearby molecules."""
        residue_molecules = {}

        # Get all CA atoms for distance calculations
        ca_atoms = []
        ca_indices = {}  # Map CA atoms to their index in c_alpha_df

        for idx, row in self.c_alpha_df.iterrows():
            for residue in Selection.unfold_entities(self.structure, "R"):
                if (residue.get_resname() != "HOH" and 'CA' in residue and
                        np.allclose([residue['CA'].get_coord()], [[row['X'], row['Y'], row['Z']]])):
                    ca_atoms.append(residue['CA'])
                    ca_indices[residue['CA']] = idx
                    residue_molecules[idx] = {
                        'residue': row['AA'],
                        'molecules': []
                    }
                    break

        ns = NeighborSearch(ca_atoms)

        # Process all non-amino molecules
        small_molecules = {
            "HOH", "P4G", "GOL", "EDO", "UDP", "GLC", "MN", "MG", "FOL", "NAG", "SO4", "ZN",
            "NAG", "CA", "DKA", "AR7", "0QE", "BMA", "D67", "CLR", "PO4","AGH",
            "POG", "1WV", "KBY", "ATP", "HIC", "UNK", "FNC", "IOD", "CNC", "CL",
            "EPE", "FUC", "RET", "MAN", "NA", "K", "CR2", "CRO", "CIT", "PEG", "ACT",
            "PDO", "PIO", "ABU", "DZP", "1IO", "LDP", "GTP", "IPA", "UPG", "XQ2",
            "CDL", "BCG", "LMT", "OCT", "1PE", "PG4", "D12", "D10", "HEX",
            "1IO", "BGC", "PLM", "NH2", "Y01", "DMS", "G3C", "LVW", "W20", "HG", "TYS", "ACE", "SEP",
            "TPO", "9EG", "43I", "IAC", "SCN", "PLM", "MYR", "PLC", "FAD", "HEM",
            "LBN", "CAC", "IMD", "GSP", "Y00", "MGO", "P2E", "IP9", "RV2", "ACY", "2CV",
            "FMT", "FVK", "NRQ", "UBL", "CU", "RR6", "ADP", "AF3", "GAL", "GLA", "5FW",
            "CFF", "TRS", "HCY", "AXL", "OLC", "UNL", "SIN", "1I8", "WZ0", "NG0", "PGE",
            "D6M", "PTY", "P42", "ACP", "LNR", "N7P", "6EA", "9PG", "BNG", "RTO", "TWT",
            "TRD", "MSE", "P0G", "FLC", "RTV", "LDP", "HSM", "PCA", "MTX", "B3P", "Y9Q",
            "CVV", "OLA", "6EA", "LPD"
        }

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in small_molecules:
                        for atom in residue:
                            coords = atom.get_coord()
                            nearest = ns.search(coords, radius=10.0)

                            if nearest:
                                distances = [np.linalg.norm(coords - ca.get_coord()) for ca in nearest]
                                closest_idx = np.argmin(distances)
                                closest_ca = nearest[closest_idx]
                                residue_idx = ca_indices[closest_ca]

                                residue_molecules[residue_idx]['molecules'].append({
                                    'type': residue.get_resname(),
                                    'coords': coords,
                                    'distance': distances[closest_idx]
                                })

        return residue_molecules

    def create_molecule_dataframe(self):
        """Create a DataFrame of all molecules with their associated residue information."""
        molecule_data = []

        for residue_idx, data in self.residue_molecules.items():
            residue_type = data['residue']
            for mol in data['molecules']:
                molecule_data.append({
                    'Residue_Index': residue_idx,
                    'Residue_Type': residue_type,
                    'Molecule_Type': mol['type'],
                    'X': mol['coords'][0],
                    'Y': mol['coords'][1],
                    'Z': mol['coords'][2],
                    'Distance': mol['distance']
                })

        return pd.DataFrame(molecule_data)

    def add_molecule_counts_to_residues(self):
        """Add molecule count information to the c_alpha_df."""
        if not self.molecule_df.empty:
            molecule_counts = self.molecule_df.groupby(
                ['Residue_Index', 'Molecule_Type']
            ).size().unstack(fill_value=0)

            self.c_alpha_df = pd.concat([self.c_alpha_df, molecule_counts], axis=1)

    def calculate_neighborhood_info(self, neighborhood_radius=5.0):
        """Calculate the neighborhood information for each C-alpha atom."""
        atom_list = [atom for atom in self.structure.get_atoms() if atom.name == 'CA']
        neighbor_search = NeighborSearch(atom_list)
        neighborhood_info = []

        for res in self.structure.get_residues():
            if 'CA' in res:
                center = res['CA'].get_coord()
                neighbors = neighbor_search.search(center, neighborhood_radius)
                distances = [np.linalg.norm(center - neighbor.get_coord()) for neighbor in neighbors if neighbor != res['CA']]
                avg_distance = np.mean(distances) if distances else 0
                max_distance = np.max(distances) if distances else 0
                count_neighbors = len(distances)
                neighborhood_info.append([avg_distance, max_distance, count_neighbors])
        return neighborhood_info

    def encode_amino_acid_properties(self, aa_letters):
        """Encode the amino acid properties for the given amino acid letters."""
        encoded_features = []
        for aa in aa_letters:
            aa_str = aa[0]  # Convert numpy array to string
            if aa_str in self.aa_info_dict_short:
                avg_mass = self.aa_info_dict_short[aa_str]['Avg. mass (Da)']
                encoded_features.append([aa_str, avg_mass])
            else:
                encoded_features.append([aa_str, 0])
        return np.array(encoded_features)

    def prepare_autoencoder_input(self):
        """Prepare the input data for the autoencoder with additional molecule features."""
        print("\nDebug: Preparing autoencoder input...")

        # Get base features
        coords = self.c_alpha_df[['X', 'Y', 'Z']].values
        aa_letters = self.c_alpha_df['AA'].values.reshape(-1, 1)
        encoded_features = self.encode_amino_acid_properties(aa_letters)
        neighborhood_info = self.calculate_neighborhood_info()

        # Get unique molecule types and print them
        molecule_types = sorted(self.molecule_df['Molecule_Type'].unique()) if not self.molecule_df.empty else []
        print(f"\nDebug: Found molecule types: {molecule_types}")

        # Create molecule features for each residue
        molecule_features = []

        print("\nDebug: Creating features for each molecule type...")
        for idx in range(len(self.c_alpha_df)):
            residue_features = []

            for mol_type in molecule_types:
                # Get molecules of this type near this residue
                mol_data = self.molecule_df[
                    (self.molecule_df['Residue_Index'] == idx) &
                    (self.molecule_df['Molecule_Type'] == mol_type)
                    ]

                if not mol_data.empty:
                    count = len(mol_data)
                    avg_distance = mol_data['Distance'].mean()
                    min_distance = mol_data['Distance'].min()
                    max_distance = mol_data['Distance'].max()
                    if idx == 0:  # Print example for first residue
                        print(f"Debug: For molecule {mol_type} at residue 0:")
                        print(f"       Count: {count}, Avg dist: {avg_distance:.2f}")
                else:
                    count, avg_distance, min_distance, max_distance = 0, -1, -1, -1

                residue_features.extend([count, avg_distance, min_distance, max_distance])

            molecule_features.append(residue_features)

        # Convert to numpy array
        molecule_features = np.array(molecule_features) if molecule_features else np.empty((len(self.c_alpha_df), 0))

        # Create column names for molecule features
        molecule_feature_columns = []
        for mol_type in molecule_types:
            new_columns = [
                f'{mol_type}_count',
                f'{mol_type}_avg_dist',
                f'{mol_type}_min_dist',
                f'{mol_type}_max_dist'
            ]
            molecule_feature_columns.extend(new_columns)
            print(f"\nDebug: Adding columns for {mol_type}:")
            print(f"       {new_columns}")

        # Combine all features
        features_to_combine = [coords, encoded_features, np.array(neighborhood_info)]
        if len(molecule_features) > 0:
            features_to_combine.append(molecule_features)

        all_features = np.hstack(features_to_combine)

        # Create column names
        columns = (
                ['X', 'Y', 'Z'] +  # Coordinates
                ['AA', 'Avg_Mass'] +  # AA properties
                ['Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count'] +  # Neighborhood info
                molecule_feature_columns  # Molecule features
        )

        print(f"\nDebug: Final number of features: {len(columns)}")
        print(f"Debug: Molecule-specific features: {len(molecule_feature_columns)}")

        return pd.DataFrame(all_features, columns=columns)
    def create_graph_with_molecules(self, distance_threshold=5.0):
        """Create a graph with both amino acids and molecule information."""
        # Get the enhanced features from autoencoder input
        node_features_df = self.prepare_autoencoder_input()

        # Label encode the AA column
        le = LabelEncoder()
        node_features_df['AA_encoded'] = le.fit_transform(node_features_df['AA'])

        # Convert all features to float
        feature_columns = [col for col in node_features_df.columns
                           if col not in ['AA']]  # Exclude non-numeric columns
        node_features = torch.tensor(
            node_features_df[feature_columns].values,
            dtype=torch.float
        )

        # Calculate edges based on distance
        coords = node_features_df[['X', 'Y', 'Z']].values
        edge_index = []
        edge_features = []

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= distance_threshold:
                    # Add bidirectional edges
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([dist, dist])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)

        # Create the Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            aa_labels=node_features_df['AA'].values
        )

        # Store feature names as an attribute
        data.feature_names = feature_columns

        return data, le