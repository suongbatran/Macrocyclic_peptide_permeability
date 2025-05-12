import numpy as np
import pandas as pd
import random
import rdkit
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

# Citiation for smiles_to_graph
# @misc{gnns_for_chemists,
#   author = {Fooladi, Hosein},
#   title = {GNNs For Chemists: Implementations of Graph Neural Networks from Scratch for Chemical Applications},
#   year = {2025},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/HFooladi/GNNs-For-Chemists}},
#   note = {Educational resource for chemists, pharmacists, and researchers interested in applying Graph Neural Networks to chemical problems}
# }
def smiles_to_graph(smiles):
    """
    Convert a SMILES string to graph representation with the following features.

    Atom Features Now Include:
        Element type 
        Formal charge 
        Hybridization state 
        Aromaticity 
        Ring membership
        Atomic degree (number of bonds)
        Hydrogen count
    Bond Features Now Include:
        Bond type 
        Conjugation 
        Ring membership 

    Args:
        smiles (str): SMILES string of the molecule

    Returns:
        tuple: Node features, adjacency matrix, edge features, edge indices
    """
    # Create RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add hydrogens to explicit representation
    mol = Chem.AddHs(mol)

    # Map bond types to indices
    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    # Get the number of atoms
    n_atoms = mol.GetNumAtoms()

    # Advanced node features
    node_features = []
    for atom in mol.GetAtoms():
        # Basic properties
        atom_type = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())

        # Create one-hot encoding for atom type (C, O, N, H, F, P, S, Cl, Br, I, or other)
        atom_types = ['C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]
        if atom_type not in atom_types:
            atom_type_onehot.append(1)  # "Other" category
        else:
            atom_type_onehot.append(0)

        # One-hot for hybridization
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]
        hybridization_onehot = [1 if hybridization == h else 0 for h in hybridization_types]
        if hybridization not in hybridization_types:
            hybridization_onehot.append(1)  # "Other" hybridization
        else:
            hybridization_onehot.append(0)

        # Combine all features
        features = atom_type_onehot + [
            formal_charge,
            is_aromatic,
            is_in_ring,
            atom.GetDegree(),         # Number of directly bonded neighbors
            atom.GetTotalNumHs(),     # Total number of Hs (explicit and implicit)
            atom.GetNumRadicalElectrons()  # Number of radical electrons
        ] + hybridization_onehot

        node_features.append(features)

    # Convert to numpy array
    node_features = np.array(node_features)

    # Create adjacency matrix and edge features
    adjacency = np.zeros((n_atoms, n_atoms))
    edge_features = []
    edge_indices = []

    for bond in mol.GetBonds():
        # Get the atoms in the bond
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Update adjacency matrix (symmetric)
        adjacency[begin_idx, end_idx] = 1
        adjacency[end_idx, begin_idx] = 1

        # Advanced bond features
        bond_type = bond.GetBondType()
        bond_type_onehot = np.zeros(len(bond_type_to_idx))
        if bond_type in bond_type_to_idx:
            bond_type_onehot[bond_type_to_idx[bond_type]] = 1

        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())

        # Combine all bond features
        features = np.concatenate([bond_type_onehot, [is_conjugated, is_in_ring]])

        # Add edge in both directions (undirected graph)
        edge_features.append(features)
        edge_indices.append((begin_idx, end_idx))

        edge_features.append(features)  # Same feature for the reverse direction
        edge_indices.append((end_idx, begin_idx))

    # Convert edge features to numpy array
    if edge_features:
        edge_features = np.array(edge_features)
    else:
        edge_features = np.empty((0, len(bond_type_to_idx) + 2))  # +2 for conjugation and ring

    return node_features, adjacency, edge_features, edge_indices

def mol_to_graph(mol, conf, group_id, conf_energy=None):
    """
    Parameters
    ----------
    mol : RDKit Mol        (with hydrogens!)
    conf: RDKit Conformer  (one of mol.GetConformer(i))
    group_id : int         identifier of the parent molecule
    conf_energy : float or None

    Returns
    -------
    torch_geometric.data.Data with:
        • x   : categorical/chemical features (same as before)
        • z   : atomic numbers  (needed by SchNet)
        • pos : [N,3] positions from the conformer
        • edge_index / edge_attr : as before
        • mol_id : group_id  (for later aggregation)
        • conf_energy (optional) : used by softmax / softmin agg
    """
    # node/bond categorical features 
    node_f, _, edge_f, edge_pairs = smiles_to_graph(Chem.MolToSmiles(mol, isomericSmiles=True))

    # coordinates 
    xyz = np.array(conf.GetPositions(), dtype=np.float32)   # shape [N,3]
    z   = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)

    data = Data(
        x   = torch.tensor(node_f,  dtype=torch.float),
        z   = torch.tensor(z,       dtype=torch.long),
        pos = torch.tensor(xyz,     dtype=torch.float),
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(edge_f,     dtype=torch.float),
        mol_id     = torch.tensor([group_id], dtype=torch.long).repeat(z.size)  # broadcast later
    )
    if conf_energy is not None:
        data.conf_energy = torch.tensor([conf_energy], dtype=torch.float).repeat(z.size)
    return data

def load_smiles(seq_pickle_path):
    with open(seq_pickle_path, "rb") as f:
        data = pickle.load(f)
    smiles = data.get("smiles")
    return smiles

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)