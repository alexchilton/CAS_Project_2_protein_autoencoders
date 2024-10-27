import os
import random
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import glob
import Bio

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.internal_coords import *
from Bio.PDB.PICIO import write_PIC, read_PIC, read_PIC_seq
from Bio.PDB.ic_rebuild import write_PDB, IC_duplicate, structure_rebuild_test
from Bio.PDB.SCADIO import write_SCAD
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import NeighborSearch

import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

parser=PDBParser(PERMISSIVE=1)
aa_info=pd.read_csv('aa_mass_letter.csv')
aa_info_dict = aa_info.set_index('Abbrev.').to_dict(orient='index')

def process_protein(file_path):
    """
    Given the file path the protein structures are retrieved from the pdf file,
   get alpha crabon info and 3D coordinated, map with amminoacid name, get heighborhood info, 
   return alpha carbon matrix with aa info and neighborhood for autoencoder input
    """

    # Parse the PDB file
    structure = parser.get_structure("protein", file_path)
    
    # Extract C-alpha information
    c_alpha_matrix = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().title()

                # Check if residue name is in aa_info dictionary
                if res_name in aa_info_dict:
                    aa_letter = aa_info_dict[res_name]['Short']
                    # Check if residue has a C-alpha atom
                    if 'CA' in residue:
                        # Get C-alpha coordinates
                        ca_atom = residue['CA']
                        x, y, z = ca_atom.get_coord()
                    
                        # Append to the matrix
                        c_alpha_matrix.append([
                            x, y, z,                         # 3D coordinates
                            aa_letter,                       # Amino acid letter
                            aa_info_dict[res_name]['Avg. mass (Da)']  # Mass info
                        ])
                    else:
                        print(f"No C-alpha atom found for residue {res_name} in chain {chain.id}")
                else:
                    print(f"Residue {res_name} not in amino acid info dictionary.")

    # Convert C-alpha information to DataFrame
    c_alpha_df = pd.DataFrame(c_alpha_matrix, columns=['X', 'Y', 'Z', 'AA', 'Mass'])
    
    # If no C-alpha atoms are found, skip the file
    if c_alpha_df.empty:
        print(f"No C-alpha atoms found in {file_path}")
        return None
    
    # Step 2: Use dictionary-based encoding (just average mass for simplicity)
    encoded_features = np.array(c_alpha_df['Mass']).reshape(-1, 1)

    # Step 3: Compute neighborhood information
    atom_list = [atom for atom in structure.get_atoms() if atom.name.strip() == 'CA']  # Use only C-alpha atoms
    neighbor_search = NeighborSearch(atom_list)
    
    neighborhood_info = []
    for res in structure.get_residues():
        if 'CA' in res:
            center = res['CA'].get_coord()
            neighbors = neighbor_search.search(center, 5.0)  # Neighborhood radius of 5 Å
            distances = [np.linalg.norm(center - neighbor.get_coord()) for neighbor in neighbors if neighbor != res['CA']]
            
            avg_distance = np.mean(distances) if distances else 0
            max_distance = np.max(distances) if distances else 0
            count_neighbors = len(distances)
            
            neighborhood_info.append([avg_distance, max_distance, count_neighbors])
    
    # Step 4: Combine all features into the matrix for autoencoder input
    coords = c_alpha_df[['X', 'Y', 'Z']].values
    autoencoder_input = np.hstack([coords, encoded_features, np.array(neighborhood_info)])
    
    return autoencoder_input
