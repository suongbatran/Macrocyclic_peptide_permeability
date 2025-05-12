import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import torch.nn as nn
import random
import torch

def load_smiles(seq_pickle_path):
    with open(seq_pickle_path, "rb") as f:
        data = pickle.load(f)
    smiles = data.get("smiles")
    return smiles

def compute_morganfp_from_smiles(smiles, radius=2, n_bits=2048):
    """
    Compute a Morgan fingerprint bit vector from a SMILES string and return as a torch.FloatTensor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bit_vect, arr)
    return torch.from_numpy(arr).float()

