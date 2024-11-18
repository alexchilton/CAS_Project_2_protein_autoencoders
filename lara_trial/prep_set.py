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

class ProteinDataset:
    def __init__(self):
        self.proteins = []

    def add_protein(self, df: pd.DataFrame, pdb_id: str, molecules_present: set):
        self.proteins.append({
            'data': df,
            'pdb_id': pdb_id,
            'molecules': molecules_present,
            'shape': df.shape
        })

    def get_protein_info(self):
        for protein in self.proteins:
            print(f"\nPDB ID: {protein['pdb_id']}")
            print(f"Shape: {protein['shape']}")
            print(f"Molecules present: {protein['molecules']}")

    def get_proteins_with_molecule(self, molecule_type: str):
        return [p for p in self.proteins if molecule_type in p['molecules']]

    def get_common_molecules(self):
        if not self.proteins:
            return set()
        common_mols = self.proteins[0]['molecules']
        for protein in self.proteins[1:]:
            common_mols = common_mols.intersection(protein['molecules'])
        return common_mols

