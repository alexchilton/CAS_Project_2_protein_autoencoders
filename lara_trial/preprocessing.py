import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import AminoAcidEncoder
from tqdm import tqdm 

class AminoAcidEncoder:
    """Enhanced amino acid encoder incorporating physical and chemical properties."""

    def __init__(self):
        self.aa_to_int = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
        self.properties = {
            'A': [0.616, -0.733, 0.0, -0.465, -1.0],  # Alanine
            'R': [-1.0, 0.667, 1.0, 1.0, 0.324],  # Arginine
            'N': [-0.524, 0.0, 0.0, 0.134, -0.262],  # Asparagine
            'D': [-0.262, 0.0, -1.0, 1.0, -0.262],  # Aspartic acid
            'C': [0.846, -0.733, 0.0, -0.465, -0.262],  # Cysteine
            'Q': [-0.524, 0.0, 0.0, 0.134, 0.324],  # Glutamine
            'E': [-0.262, 0.0, -1.0, 1.0, 0.324],  # Glutamic acid
            'G': [0.616, -1.0, 0.0, -0.465, -1.0],  # Glycine
            'H': [-0.524, 0.667, 0.5, 0.134, 0.324],  # Histidine
            'I': [1.0, 0.667, 0.0, -1.0, 0.324],  # Isoleucine
            'L': [1.0, 0.667, 0.0, -1.0, 0.324],  # Leucine
            'K': [-1.0, 0.667, 1.0, 1.0, 0.324],  # Lysine
            'M': [0.846, 0.667, 0.0, -0.465, 0.324],  # Methionine
            'F': [1.0, 1.0, 0.0, -1.0, 1.0],  # Phenylalanine
            'P': [0.0, -0.733, 0.0, -0.465, -0.262],  # Proline
            'S': [0.0, -0.733, 0.0, 0.134, -0.262],  # Serine
            'T': [0.0, -0.733, 0.0, 0.134, -0.262],  # Threonine
            'W': [0.846, 1.0, 0.0, -0.465, 1.0],  # Tryptophan
            'Y': [0.846, 1.0, 0.0, 0.134, 1.0],  # Tyrosine
            'V': [1.0, 0.0, 0.0, -1.0, 0.324]  # Valine
        }

    def encode(self, aa):
        """Encode a single amino acid."""
        onehot = np.zeros(len(self.aa_to_int))
        properties = np.zeros(5)  # Default for unknown residues

        if aa in self.aa_to_int:
            onehot[self.aa_to_int[aa]] = 1
            properties = self.properties.get(aa, properties)

        return np.concatenate([onehot, properties])


class ProteinAnalyzer:
    def __init__(self, pdb_directory, aa_info_file):
        self.pdb_directory = pdb_directory
        self.aa_info_file = aa_info_file
        self.aa_info = None
        self.parser = PDBParser(QUIET=True)
        self.encoder = AminoAcidEncoder()
        self.three_to_one = {  # 3-letter to 1-letter mapping
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
        }
        self.non_amino_acids = {
            "HOH", "P4G", "GOL", "EDO", "UDP", "GLC", "MN", "MG", "FOL", "NAG", "SO4", "ZN",
            "NAG", "CA", "DKA", "AR7", "0QE", "BMA", "D67", "CLR", "PO4", "AGH",
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

    def load_aa_info(self):
        """Load amino acid information from the CSV file."""
        self.aa_info = pd.read_csv(self.aa_info_file)
        self.aa_info.set_index("Short", inplace=True)
        print("Loaded amino acid information.")

    def parse_pdb(self, pdb_file):
        """Parse a PDB file to extract the protein structure."""
        structure_id = os.path.basename(pdb_file).split('.')[0]
        try:
            structure = self.parser.get_structure(structure_id, pdb_file)
            #print(f"Parsed PDB file: {pdb_file}")
            return structure
        except Exception as e:
            print(f"Error parsing PDB file {pdb_file}: {e}")
            return None

    def extract_features(self, structure):
        """
        Extract features from a protein structure, including encoding and neighborhood metrics.
        """
        if structure is None:
            return []

        residues = []
        coordinates = []
        non_amino_acid_residues = set()  # Track non-amino acid residues

        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()

                    # Skip non-amino acid residues dynamically or from the list
                    if resname in self.non_amino_acids or "CA" not in residue:
                        non_amino_acid_residues.add(resname)
                        continue

                    # Convert 3-letter to 1-letter code
                    aa_short = self._resname_to_short(resname)
                    if not aa_short:
                        continue  # Skip unknown residues

                    coord = residue["CA"].get_coord()
                    encoding = self.encoder.encode(aa_short)

                    residues.append({
                        "name": resname,
                        "short": aa_short,
                        "encoding": encoding,
                        "coord": coord
                    })
                    coordinates.append(coord)

        # Compute neighborhood metrics
        coordinates = np.array(coordinates)
        if len(coordinates) > 0:
            avg_dist, max_dist, neighbor_count = self._compute_neighborhood_metrics(coordinates)
            for idx, residue in enumerate(residues):
                residue["Avg_Neighbor_Dist"] = avg_dist[idx]
                residue["Max_Neighbor_Dist"] = max_dist[idx]
                residue["Neighbor_Count"] = neighbor_count[idx]

        # Log non-amino acid residues
        #if non_amino_acid_residues:
        #    print(f"Non-amino acid residues encountered: {non_amino_acid_residues}")

        #print(f"Extracted {len(residues)} residues with encoding and neighborhood metrics.")
        return residues

    def prepare_data(self):
        """
        Prepare data for the autoencoder by processing all PDB files in the directory.
        Returns a list of residue arrays, one for each PDB file, with a clean progress bar.
        """
        all_pdb_features = []
        pdb_files = [f for f in os.listdir(self.pdb_directory) if f.endswith(".pdb")]

        print(f"Processing {len(pdb_files)} PDB files...")
        with tqdm(pdb_files, desc="Processing PDB files", leave=True) as progress_bar:
            for pdb_file in progress_bar:
                structure = self.parse_pdb(os.path.join(self.pdb_directory, pdb_file))
                features = self.extract_features(structure)
                all_pdb_features.append(features)

                # Update the progress bar description dynamically
                progress_bar.set_postfix({"Processed": len(all_pdb_features)})

        print(f"Prepared data for {len(all_pdb_features)} PDB files.")
        return all_pdb_features


    def _resname_to_short(self, resname):
        """Convert 3-letter residue name to 1-letter code."""
        mapped_resname = self.three_to_one.get(resname)
        #if mapped_resname is None:
        #    print(f"Warning: Residue '{resname}' is not a recognized amino acid.")
        return mapped_resname

    def _compute_neighborhood_metrics(self, coordinates):
        """
        Compute neighborhood metrics: average distance, max distance, and neighbor count.
        """
        avg_distances = []
        max_distances = []
        neighbor_counts = []

        for i in range(len(coordinates)):
            distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
            distances = distances[distances > 0]  # Exclude self-distance
            avg_distances.append(np.mean(distances) if len(distances) > 0 else 0)
            max_distances.append(np.max(distances) if len(distances) > 0 else 0)
            neighbor_counts.append(len(distances))

        return avg_distances, max_distances, neighbor_counts
