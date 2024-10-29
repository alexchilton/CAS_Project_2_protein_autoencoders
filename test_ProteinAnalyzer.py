import pytest
import pandas as pd
import numpy as np
from ProteinAnalyzer import ProteinAnalyzer
import Bio

@pytest.fixture
def protein_analyzer():
    pdb_file = 'test.pdb'
    aa_info_file = 'aa_mass_letter.csv'
    return ProteinAnalyzer(pdb_file, aa_info_file)

def test_initialization(protein_analyzer):
    assert protein_analyzer.pdb_file == 'test.pdb'
    assert protein_analyzer.aa_info_file == 'aa_mass_letter.csv'
    assert isinstance(protein_analyzer.structure, Bio.PDB.Structure.Structure)
    assert isinstance(protein_analyzer.aa_info_dict, dict)
    assert isinstance(protein_analyzer.aa_info_dict_short, dict)

def test_extract_c_alpha_info(protein_analyzer):
    c_alpha_df = protein_analyzer.extract_c_alpha_info()
    assert isinstance(c_alpha_df, pd.DataFrame)
    assert not c_alpha_df.empty
    assert 'X' in c_alpha_df.columns
    assert 'Y' in c_alpha_df.columns
    assert 'Z' in c_alpha_df.columns
    assert 'AA' in c_alpha_df.columns
    assert 'Mass' in c_alpha_df.columns

def test_prepare_autoencoder_input(protein_analyzer):
    autoencoder_input_df = protein_analyzer.prepare_autoencoder_input()
    assert isinstance(autoencoder_input_df, pd.DataFrame)
    assert not autoencoder_input_df.empty
    assert autoencoder_input_df.shape[1] == 7
    expected_columns = ['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count']
    assert list(autoencoder_input_df.columns) == expected_columns


def test_calculate_pairwise_distances(protein_analyzer):
    coords = np.array([[0, 0, 0], [3, 4, 0]])
    distance_matrix = protein_analyzer.calculate_pairwise_distances(coords)
    assert distance_matrix.shape == (2, 2)
    assert distance_matrix[0, 1] == 5.0
    assert distance_matrix[1, 0] == 5.0

def test_encode_amino_acid_properties(protein_analyzer):
    aa_letters = ['A', 'C', 'D']
    encoded_features = protein_analyzer.encode_amino_acid_properties(aa_letters)
    assert encoded_features.shape == (3, 1)
    assert encoded_features[0, 0] == protein_analyzer.aa_info_dict_short['A']['Avg. mass (Da)']

def test_calculate_neighborhood_info(protein_analyzer):
    neighborhood_info = protein_analyzer.calculate_neighborhood_info()
    assert isinstance(neighborhood_info, list)
    assert len(neighborhood_info) > 0
    assert len(neighborhood_info[0]) == 3

def test_pad_dataframe(protein_analyzer):
    df = pd.DataFrame(np.ones((5, 7)), columns=['X', 'Y', 'Z', 'Avg_Mass', 'Avg_Neighbor_Dist', 'Max_Neighbor_Dist', 'Neighbor_Count'])
    target_shape = (10, 7)
    padded_df = protein_analyzer.pad_dataframe(df, target_shape)
    assert padded_df.shape == target_shape
    assert (padded_df.iloc[5:].values == 0).all()