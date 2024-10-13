import os
import numpy as np
from Bio import PDB

def load_pdb_files(directory):
    parser = PDB.PDBParser()
    pdb_files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
    pdb_files.sort(key=lambda f: os.path.getsize(os.path.join(directory, f)))

    for pdb_file in pdb_files:
        file_path = os.path.join(directory, pdb_file)
        print(f"Opening file: {file_path} (Size: {os.path.getsize(file_path)} bytes)")
        structure = parser.get_structure(pdb_file, file_path)
        yield structure, pdb_file

def calculate_distance_matrix(structure):
    model = structure[0]
    chain = model.child_list[0]
    num_residues = len(chain)
    distance_matrix = np.zeros((num_residues, num_residues))

    for i, residue1 in enumerate(chain):
        for j, residue2 in enumerate(chain):
            distance_matrix[i, j] = residue1['CA'] - residue2['CA']

    print(f"Distance matrix size: {distance_matrix.shape}")
    return distance_matrix

def save_pdb_file(structure, directory, filename):
    io = PDB.PDBIO()
    io.set_structure(structure)
    file_path = os.path.join(directory, filename)
    print(f"Saving file: {file_path}")
    io.save(file_path)

# Directories
dir_x = '/Users/alexchilton/Downloads/archive/train'
dir_y = '/Users/alexchilton/Downloads/archive/just10000'

i = 0
# Load and filter structures
filtered_structures = []
for structure, pdb_file in load_pdb_files(dir_x):
    distance_matrix = calculate_distance_matrix(structure)
    if distance_matrix.shape[0] < 250:
        filtered_structures.append((structure, pdb_file))
        i = i + 1
        print(f"Filtered structures: {i}")
    if len(filtered_structures) >= 5000:
        break

# Save filtered structures
for structure, pdb_file in filtered_structures:
    save_pdb_file(structure, dir_y, pdb_file)