import pickle
import e3fp
from e3fp.pipeline import fprints_from_mol

import numpy as np
import torch.nn as nn
import random
import torch
from torch_geometric.data import Data

def mol_to_e3fp(
    mol,
    conf, bits, level, radius_multiplier,
    group_id,
    conf_energy=None,
):
    """
    Parameters
    ----------
    mol : RDKit Mol        (with hydrogens!)
    conf : RDKit Conformer (one of mol.GetConformer(i))
    group_id : int         identifier of the parent molecule
    conf_energy : float or None

    bits : int, optional
        Length of the folded fingerprint (default: 2048)
    level : int, optional
        Max number of shell‐expansion iterations (default: 3)
    radius_multiplier : float, optional
        Shell radius increment multiplier (default: 1.5)

    Returns
    -------
    torch_geometric.data.Data with fields:
        e3fp        : torch.FloatTensor, shape (bits,)
        mol_id      : torch.LongTensor, shape (1,)
        conf_energy : torch.FloatTensor, shape (1,) (only if not None)
    """
    # 1) Generate E3FPs for all conformers in the mol
    fprint_params = {
        'bits': bits,
        'level': level,
        'radius_multiplier': radius_multiplier,
        'first': -1
    }
    fprints = fprints_from_mol(mol, fprint_params=fprint_params)

    # 2) Extract this conformer's fingerprint
    fp = fprints[0]

    # 3) Get a dense NumPy vector and convert to torch.FloatTensor
    arr = fp.to_vector(sparse=False).astype(np.float32)
    e3fp_tensor = torch.from_numpy(arr)

    # 4) Build the PyG Data object
    data = Data(e3fp=e3fp_tensor,
                mol_id=torch.tensor([group_id], dtype=torch.long))
    if conf_energy is not None:
        data.conf_energy = torch.tensor([conf_energy], dtype=torch.float)

    return data