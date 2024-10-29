import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import Bio

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.internal_coords import *
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import parse_pdb_header
from Bio.PDB import Selection




import matplotlib.lines as mlines
from jinja2.async_utils import auto_to_list
from mpl_toolkits.mplot3d import Axes3D

import protein_parser
from protein_parser import aa_info_dict

class ProteinAnalyzer:
    def __init__(self, pdb_file, aa_info_file):
        self.pdb_file = pdb_file
        self.aa_info_file = aa_info_file
        self.parser = PDBParser(PERMISSIVE=1)
        self.structure = self.parser.get_structure('protein_structure', self.pdb_file)
        self.aa_info_dict = self.get_aa_info_dict()
        self.aa_info_dict_short = self.get_aa_info_dict_short()
        self.c_alpha_df = self.extract_c_alpha_info()

    def get_aa_info_dict(self):
        aa_info = pd.read_csv(self.aa_info_file)
        return aa_info.set_index('Abbrev.').to_dict(orient='index')

    def get_aa_info_dict_short(self):
        aa_info = pd.read_csv(self.aa_info_file)
        return aa_info.set_index('Short').to_dict(orient='index')

    def get_residue_info(self, res_name):
        if res_name in self.aa_info_dict:
            return self.aa_info_dict[res_name]['Short']
        else:
            print(f"Residue {res_name} not in amino acid info dictionary.")
            return None

    def process_residue(self, residue, res_name, c_alpha_matrix):
        if 'CA' in residue:
            ca_atom = residue['CA']
            x, y, z = ca_atom.get_coord()
            c_alpha_matrix.append([
                x, y, z,
                self.aa_info_dict[res_name]['Short'],
                self.aa_info_dict[res_name]['Avg. mass (Da)']
            ])
        else:
            print(f"No C-alpha atom found for residue {res_name} in chain {residue.parent.id}")

    def create_c_alpha_dataframe(self, c_alpha_matrix):
        c_alpha_df = pd.DataFrame(c_alpha_matrix, columns=["X", "Y", "Z", "AA", "Mass"])
        if c_alpha_df.empty:
            print("C-alpha DataFrame is empty. Check residue names or PDB file.")
        else:
            print(c_alpha_df)
        return c_alpha_df

    def extract_c_alpha_info(self):
        c_alpha_matrix = []
        res_list = Selection.unfold_entities(self.structure, "R")
        for residue in res_list:
            res_name = residue.get_resname().title()
            self.process_residue(residue, res_name, c_alpha_matrix)
        return self.create_c_alpha_dataframe(c_alpha_matrix)

    def calculate_pairwise_distances(self, coordinates):
        num_residues = len(coordinates)
        distance_matrix = np.zeros((num_residues, num_residues))
        for i in range(num_residues):
            for j in range(i, num_residues):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def encode_amino_acid_properties(self, aa_letters):
        encoded_features = []
        for aa in aa_letters:
            if aa in self.aa_info_dict_short:
                avg_mass = self.aa_info_dict_short[aa]['Avg. mass (Da)']
                encoded_features.append([avg_mass])
            else:
                encoded_features.append([0])
        return np.array(encoded_features)

    def calculate_neighborhood_info(self, neighborhood_radius=5.0):
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
        coords = self.c_alpha_df[['X', 'Y', 'Z']].values
        aa_letters = self.c_alpha_df['AA']
        encoded_features = self.encode_amino_acid_properties(aa_letters)
        neighborhood_info = self.calculate_neighborhood_info()
        autoencoder_input = np.hstack([coords, encoded_features, np.array(neighborhood_info)])
        columns = ['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']
        return pd.DataFrame(autoencoder_input, columns=columns)

    def pad_dataframe(self, df, target_shape):
        current_shape = df.shape
        if current_shape[0] >= target_shape[0] and current_shape[1] == target_shape[1]:
            return df
        rows_to_add = target_shape[0] - current_shape[0]
        if rows_to_add > 0:
            padding_df = pd.DataFrame(np.zeros((rows_to_add, target_shape[1])), columns=df.columns)
            padded_df = pd.concat([df, padding_df], ignore_index=True)
        else:
            padded_df = df
        return padded_df

pdb_file = '/Users/alexchilton/Downloads/archive/train/AF-D0ZA02-F1-model_v4.pdb'
aa_info_file = 'aa_mass_letter.csv'
analyzer = ProteinAnalyzer(pdb_file, aa_info_file)

# Generate the autoencoder input DataFrame
autoencoder_input_df = analyzer.prepare_autoencoder_input()


import os
import pandas as pd
from ProteinAnalyzer import ProteinAnalyzer

# Define the directory containing the PDB files and the amino acid information file
pdb_directory = '/Users/alexchilton/Downloads/archive/just100'

# Initialize an array to store the autoencoder input DataFrames
autoencoder_input_dfs = []

# Loop over each file in the directory
for pdb_file in os.listdir(pdb_directory):
    if pdb_file.endswith('.pdb'):
        pdb_path = os.path.join(pdb_directory, pdb_file)

        # Initialize the ProteinAnalyzer with the PDB file and amino acid information file
        analyzer = ProteinAnalyzer(pdb_path, aa_info_file)

        # Generate the autoencoder input DataFrame
        autoencoder_input_df = analyzer.prepare_autoencoder_input()

        # Add the DataFrame to the array
        autoencoder_input_dfs.append(autoencoder_input_df)

# Now autoencoder_input_dfs contains all the DataFrames
# Loop through the autoencoder input DataFrames and print their dimensions
for i, df in enumerate(autoencoder_input_dfs):
    print(f"DataFrame {i+1} dimensions: {df.shape}")
    