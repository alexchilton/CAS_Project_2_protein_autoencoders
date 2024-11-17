import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.PDBParser import PDBParser

class ProteinAnalyzer3:
    def __init__(self, pdb_file, aa_info_file):
        """
        Initialize the ProteinAnalyzer3 class.
        """
        self.pdb_file = pdb_file
        self.aa_info_file = aa_info_file
        self.parser = PDBParser(PERMISSIVE=1)
        self.structure = self.parser.get_structure('protein_structure', self.pdb_file)
        self.aa_info_dict = self.get_aa_info_dict()
        self.aa_info_dict_short = self.get_aa_info_dict_short()

        # Extract both C-alpha and water information
        self.c_alpha_df = self.extract_c_alpha_info()
        self.water_df = self.extract_water_info()
        self.combined_df = pd.concat([self.c_alpha_df, self.water_df], ignore_index=True)

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
        c_alpha_matrix = []
        res_list = Selection.unfold_entities(self.structure, "R")

        for residue in res_list:
            if residue.get_resname() != "HOH":  # Skip water molecules
                res_name = residue.get_resname().title()
                residue_data = self.process_residue(residue, res_name)
                if residue_data:
                    c_alpha_matrix.append(residue_data)

        return pd.DataFrame(
            c_alpha_matrix,
            columns=["X", "Y", "Z", "AA", "Mass"]
        )

    def extract_water_info(self):
        """Extract water molecule coordinates from the PDB file."""
        water_matrix = []

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() == "HOH":
                        for atom in residue:
                            x, y, z = atom.get_coord()
                            water_matrix.append([x, y, z, "HOH", 18.015])  # Water molecular mass

        if not water_matrix:
            return pd.DataFrame(columns=["X", "Y", "Z", "AA", "Mass"])

        return pd.DataFrame(
            water_matrix,
            columns=["X", "Y", "Z", "AA", "Mass"]
        )

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

    def prepare_autoencoder_input(self):
        """Prepare the input data for the autoencoder."""
        coords = self.c_alpha_df[['X', 'Y', 'Z']].values
        aa_letters = self.c_alpha_df['AA'].values.reshape(-1, 1)
        encoded_features = self.encode_amino_acid_properties(aa_letters)
        neighborhood_info = self.calculate_neighborhood_info()
        autoencoder_input = np.hstack([coords, encoded_features, np.array(neighborhood_info)])
        columns = ['X', 'Y', 'Z', 'AA', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']
        return pd.DataFrame(autoencoder_input, columns=columns)

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

pdb_file = '8g0w.pdb'
aa_info_file = 'aa_mass_letter.csv'

analyzer = ProteinAnalyzer3(pdb_file, aa_info_file)
autoencoder_input_df = analyzer.prepare_autoencoder_input()

pd.set_option('display.max_columns', None)
print(autoencoder_input_df.shape)
print(autoencoder_input_df.head())