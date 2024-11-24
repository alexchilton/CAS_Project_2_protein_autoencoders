from Bio.PDB import PDBParser, NeighborSearch, Selection
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class ProteinAnalyzer:
    def __init__(self, pdb_file, aa_info_dict, neighborhood_radius=5.0):
        """Initialize with the PDB file and amino acid information."""
        self.pdb_file = pdb_file
        self.aa_info_dict = aa_info_dict
        self.neighborhood_radius = neighborhood_radius

        # Parse the structure
        parser = PDBParser(PERMISSIVE=1)
        self.structure = parser.get_structure('protein_structure', self.pdb_file)

        # Placeholder for the results
        self.small_molecules = set()
        self.autoencoder_df = None

    def extract_alpha_carbon_info(self):
        """Extract 3D coordinates and amino acid information."""
        c_alpha_data = []
        atom_list = []  # For NeighborSearch

        for residue in Selection.unfold_entities(self.structure, "R"):  # All residues
            if 'CA' in residue:
                ca_atom = residue['CA']
                coords = ca_atom.get_coord()
                res_name = residue.get_resname()

                # Encode amino acid properties
                avg_mass = self.aa_info_dict.get(res_name, {}).get('Avg. mass (Da)', 0)

                c_alpha_data.append([coords[0], coords[1], coords[2], avg_mass])
                atom_list.append(ca_atom)
            else:
                # Track small molecules
                self.small_molecules.add(residue.get_resname())

        # Convert to DataFrame
        columns = ['X', 'Y', 'Z', 'Avg_Mass']
        c_alpha_df = pd.DataFrame(c_alpha_data, columns=columns)

        return c_alpha_df, atom_list

    def compute_neighborhood_info(self, c_alpha_df, atom_list):
        """Compute neighborhood statistics for each C-alpha atom."""
        # Build NeighborSearch tree
        neighbor_search = NeighborSearch(atom_list)

        neighborhood_info = []
        for idx, row in c_alpha_df.iterrows():
            center = np.array([row['X'], row['Y'], row['Z']])
            neighbors = neighbor_search.search(center, self.neighborhood_radius)

            # Exclude the current atom from neighbors
            distances = [
                np.linalg.norm(center - neighbor.get_coord())
                for neighbor in neighbors if not np.allclose(neighbor.get_coord(), center)
            ]

            # Neighborhood features
            avg_distance = np.mean(distances) if distances else 0
            max_distance = np.max(distances) if distances else 0
            count_neighbors = len(distances)

            neighborhood_info.append([avg_distance, max_distance, count_neighbors])

        # Convert to DataFrame
        neighborhood_columns = ['Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']
        neighborhood_df = pd.DataFrame(neighborhood_info, columns=neighborhood_columns)

        return neighborhood_df

    def process(self):
        """Process the PDB file to compute the required features."""
        # Step 1: Extract alpha carbon info and build atom list
        c_alpha_df, atom_list = self.extract_alpha_carbon_info()

        # Step 2: Compute neighborhood info
        neighborhood_df = self.compute_neighborhood_info(c_alpha_df, atom_list)

        # Step 3: Combine data
        self.autoencoder_df = pd.concat([c_alpha_df, neighborhood_df], axis=1)

        # Add small molecule count
        self.autoencoder_df['Small_Molecules_Found'] = len(self.small_molecules)

        return self.autoencoder_df

# Helper function to process a folder
def process_pdb_directory(pdb_directory, aa_info_dict, neighborhood_radius=5.0):
    """Process all PDB files in a directory."""
    pdb_files = [os.path.join(pdb_directory, f) for f in os.listdir(pdb_directory) if f.endswith('.pdb')]
    all_dataframes = []

    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        analyzer = ProteinAnalyzer(pdb_file, aa_info_dict, neighborhood_radius)
        df = analyzer.process()
        all_dataframes.append((pdb_file, df))

    return all_dataframes
