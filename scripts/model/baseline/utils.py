import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch.nn as nn

def load_smiles(seq_pickle_path):
    with open(seq_pickle_path, "rb") as f:
        data = pickle.load(f)
    smiles = data.get("smiles")
    return smiles

def compute_tpsa_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.TPSA(mol)



class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input_dim = 1 (2D_TPSA), output_dim = 1 (permeability)

    def forward(self, x):
        return self.linear(x)