import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.PDBParser import PDBParser
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import KDTree


# class ProteinAnalyzer3:
#     def __init__(self, pdb_file, aa_info_dict):
#         """Initialize with pre-validated amino acid info."""
#         self.pdb_file = pdb_file
#         self.parser = PDBParser(PERMISSIVE=1)
#         self.structure = self.parser.get_structure('protein_structure', self.pdb_file)
#         self.aa_info_dict = aa_info_dict['full']
#         self.aa_info_dict_short = aa_info_dict['short']

#         # Extract amino acid information first
#         self.c_alpha_df = self.extract_c_alpha_info()

#     def get_residue_info(self, res_name):
#         """Return the short name of the amino acid given the residue name."""
#         return self.aa_info_dict.get(res_name, {}).get('Short')

#     def process_residue(self, residue, res_name):
#         """Process a residue and extract the C-alpha atom coordinates."""
#         short_name = self.get_residue_info(res_name)
#         if not short_name:  # Skip if residue is not in the amino acid dictionary
#             return None

#         if 'CA' in residue:  # Check for the C-alpha atom
#             ca_atom = residue['CA']
#             x, y, z = ca_atom.get_coord()
#             return [
#                 x, y, z,
#                 short_name,
#                 self.aa_info_dict[res_name]['Avg. mass (Da)']
#             ]
#         return None

#     def extract_c_alpha_info(self):
#         """Extract the C-alpha atom coordinates and amino acid properties from the PDB file."""
#         self.small_molecules = set()  # Track unrecognized small molecules

#         c_alpha_matrix = []
#         res_list = Selection.unfold_entities(self.structure, "R")  # Flatten all residues

#         for residue in res_list:
#             res_name = residue.get_resname().title()  # Standardize residue name formatting

#             # Check if it's a known amino acid
#             residue_data = self.process_residue(residue, res_name)
#             if residue_data:
#                 c_alpha_matrix.append(residue_data)
#             else:
#                 # Treat unrecognized residues as small molecules
#                 self.small_molecules.add(res_name)

#         # Log dynamically discovered small molecules
#         if self.small_molecules:
#             print(f"Discovered small molecules: {self.small_molecules}")

#         # Create a DataFrame for C-alpha atoms
#         return pd.DataFrame(
#             c_alpha_matrix,
#             columns=["X", "Y", "Z", "AA", "Mass"]
#        )

class ProteinAnalyzer4:
    def __init__(self, pdb_file, aa_info_dict, neighborhood_radius=10.0):
        """Initialize with preloaded amino acid info and configurable radius."""
        self.pdb_file = pdb_file
        self.parser = PDBParser(PERMISSIVE=1)
        self.structure = self.parser.get_structure('protein_structure', self.pdb_file)
        self.aa_info_dict = aa_info_dict['full']
        self.aa_info_dict_short = aa_info_dict['short']
        self.neighborhood_radius = neighborhood_radius

        # Extract amino acid information
        self.c_alpha_df = self.extract_c_alpha_info()

        # Compute residue-small molecule mapping
        self.residue_molecules = self.create_residue_molecule_mapping()

    def extract_c_alpha_info(self):
        """Extract the C-alpha atom coordinates and amino acid properties."""
        self.small_molecules = set()  # Track unrecognized small molecules

        c_alpha_matrix = []
        res_list = Selection.unfold_entities(self.structure, "R")  # Flatten all residues

        for residue in res_list:
            res_name = residue.get_resname().title()  # Standardize residue name formatting
            residue_data = self.process_residue(residue, res_name)
            if residue_data:
                c_alpha_matrix.append(residue_data)
            else:
                self.small_molecules.add(res_name)  # Track small molecules dynamically

        print(f"Discovered small molecules: {self.small_molecules}")

        return pd.DataFrame(
            c_alpha_matrix,
            columns=["X", "Y", "Z", "AA", "Mass"]
        )

    def process_residue(self, residue, res_name):
        """Process a residue and extract the C-alpha atom coordinates."""
        short_name = self.aa_info_dict.get(res_name, {}).get('Short')
        if not short_name:
            return None

        if 'CA' in residue:
            ca_atom = residue['CA']
            x, y, z = ca_atom.get_coord()
            return [
                x, y, z,
                short_name,
                self.aa_info_dict[res_name]['Avg. mass (Da)']
            ]
        return None

    def create_residue_molecule_mapping(self):
        """Create a mapping of residues to their neighborhood information."""
        residue_molecules = {}
        ca_atoms = []
        ca_indices = {}

        # Step 1: Collect C-alpha atoms and index them
        for idx, row in self.c_alpha_df.iterrows():
            for residue in Selection.unfold_entities(self.structure, "R"):
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    if np.allclose(ca_atom.get_coord(), [row['X'], row['Y'], row['Z']]):
                        ca_atoms.append(ca_atom)
                        ca_indices[residue.get_id()] = idx
                        residue_molecules[idx] = {
                            'residue': row['AA'],
                            'avg_distance': 0,
                            'max_distance': 0,
                            'count_neighbors': 0,
                            'molecules': []
                        }
                        break

        # Build KDTree for efficient neighbor search
        ca_coords = np.array([atom.get_coord() for atom in ca_atoms])
        tree = KDTree(ca_coords)

        # Step 2: Process small molecules and compute neighborhood statistics
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in self.small_molecules:
                        for atom in residue:
                            coords = atom.get_coord()
                            
                            # Skip if coordinates are missing
                            if coords is None or not np.isfinite(coords).all():
                                print(f"Skipping molecule {residue.get_resname()} due to invalid coords")
                            continue

                            neighbors = tree.query_ball_point(coords, self.neighborhood_radius)

                            # Skip if no neighbors --> new piece added
                            if not neighbors:  
                                continue

                            # from here still old part <-----
                            distances = [
                                np.linalg.norm(coords - ca_atoms[neighbor_idx].get_coord())
                                for neighbor_idx in neighbors
                            ]

                            # Skip molecules with no valid distances--> new piece added
                            if not distances:
                                continue

                            # from here still old part <-----
                            #distances:
                            avg_distance = np.mean(distances)
                            max_distance = np.max(distances)
                            count_neighbors = len(distances)

                            for neighbor_idx in neighbors:
                                residue_idx = ca_indices.get(ca_atoms[neighbor_idx].get_parent().get_id(), None)

                                if residue_idx is not None:
                                    residue_molecules[residue_idx]['avg_distance'] += avg_distance
                                    residue_molecules[residue_idx]['max_distance'] = max(
                                        residue_molecules[residue_idx]['max_distance'],
                                        max_distance
                                    )
                                    residue_molecules[residue_idx]['count_neighbors'] += count_neighbors
                                    residue_molecules[residue_idx]['molecules'].append({
                                        'type': residue.get_resname(),
                                        'avg_distance': avg_distance,
                                        'max_distance': max_distance,
                                        'count_neighbors': count_neighbors
                                    })

        # Normalize avg_distance by dividing by the count of neighbors
        for idx, data in residue_molecules.items():
            if data['count_neighbors'] > 0:
                data['avg_distance'] /= data['count_neighbors']

        # Debugging: Print a sample of the mapping
        #print(f"Debug: Residue molecules mapping (sample): {list(residue_molecules.items())[:5]}")

        return residue_molecules



    



    # def create_residue_molecule_mapping(self):
    #     """Create a mapping of residues to their nearby small molecules."""
    #     residue_molecules = {}
    #     ca_atoms = []
    #     ca_indices = {}

    #     for idx, row in self.c_alpha_df.iterrows():
    #         for residue in Selection.unfold_entities(self.structure, "R"):
    #             if 'CA' in residue:
    #                 ca_atom = residue['CA']
    #                 if np.allclose(ca_atom.get_coord(), [row['X'], row['Y'], row['Z']]):
    #                     ca_atoms.append(ca_atom)
    #                     ca_indices[residue.get_id()] = idx
    #                     residue_molecules[idx] = {
    #                         'residue': row['AA'],
    #                         'molecules': []
    #                     }
    #                     break

    #     ns = NeighborSearch(ca_atoms)

    #     # Process small molecules
    #     for model in self.structure:
    #         for chain in model:
    #             for residue in chain:
    #                 if residue.get_resname() in self.small_molecules:
    #                     for atom in residue:
    #                         coords = atom.get_coord()
    #                         nearest = ns.search(coords, radius=self.neighborhood_radius)

    #                         ca_neighbors = [ca for ca in nearest if ca in ca_atoms]

    #                         for ca in ca_neighbors:
    #                             residue_idx = ca_indices.get(ca.get_parent().get_id(), None)

    #                             if residue_idx is not None:
    #                                 distance = np.linalg.norm(coords - ca.get_coord())
    #                                 residue_molecules[residue_idx]['molecules'].append({
    #                                     'type': residue.get_resname(),
    #                                     'coords': coords,
    #                                     'distance': distance
    #                                 })

    #     return residue_molecules

    def create_combined_dataframe(self):
        """Combine amino acid, molecule, and neighborhood features into a DataFrame."""
        # Base DataFrame
        df = self.c_alpha_df.copy()

        # Add molecule counts for each residue
        molecule_data = []
        
        for idx, data in self.residue_molecules.items():
            for mol in data['molecules']:
                molecule_data.append({
                    'Residue_Index': idx,
                    'Molecule_Type': mol['type'],
                    'X': mol['coords'][0],
                    'Y': mol['coords'][1],
                    'Z': mol['coords'][2],
                    'Distance': mol['distance']
                })

        # Create molecule DataFrame
        molecule_df = pd.DataFrame(molecule_data)
        if not molecule_df.empty:
            molecule_counts = molecule_df.groupby(
                ['Residue_Index', 'Molecule_Type']
            ).size().unstack(fill_value=0)

            # Merge molecule counts into the main DataFrame
            df = pd.concat([df, molecule_counts], axis=1)

        # Add neighborhood statistics to the DataFrame
        df['Avg_Neighbor_Distance'] = df.index.map(
        lambda idx: self.residue_molecules[idx]['avg_distance']
        )
        df['Max_Neighbor_Distance'] = df.index.map(
            lambda idx: self.residue_molecules[idx]['max_distance']
        )
        df['Neighbor_Count'] = df.index.map(
            lambda idx: self.residue_molecules[idx]['count_neighbors']
        )

        return df


