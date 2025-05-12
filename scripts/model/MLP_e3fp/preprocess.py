import os, pickle
import pandas as pd
import argparse
from pathos.multiprocessing import ProcessingPool as Pool
import logging
import time
import torch
from tqdm import tqdm
import e3fp
from utils import mol_to_e3fp
from rdkit import Chem, RDLogger
import heapq, argparse
import torch.multiprocessing as mp
from torch_geometric.data import InMemoryDataset

RDLogger.DisableLog('rdApp.*')
mp.set_sharing_strategy('file_system')


def process_single_row(args):
    """
    returns *list* of Data objects (one per conformer) so that Pool-imap
    keeps the original API.
    """
    
    idx, row, pickle_dir, bits, level, radius = args
    seq, perm = row["sequence"], row["permeability"]

    pkl = os.path.join(pickle_dir, f"{seq}.pickle")
    if not os.path.exists(pkl):
        logging.error(f"Pickle missing for {seq}")
        return []

    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
        mol        = d["rd_mol"]

    conf_dicts = d.get("conformers", [])
    if not conf_dicts:
        logging.warning(f"No conformer metadata for {seq}")
        return []

    # build (energy, mol_conformer_idx) pairs
    pairs = []
    for conf in conf_dicts:
        set_id = conf.get("set")
        if set_id is None:
            continue
        mol_idx = set_id - 1
        if not (0 <= mol_idx < mol.GetNumConformers()):
            logging.warning(f"Bad set index {set_id} for {seq}, skipping")
            continue
        energy = conf.get("relativeenergy", None)
        if energy is None:
            logging.warning(f"Missing relativeenergy for conformer {set_id} of {seq}")
            continue
        pairs.append((energy, mol_idx))

    # pick the 50 lowest-energy conformers
    top50 = heapq.nsmallest(50, pairs, key=lambda x: x[0])
    conf_ids = [mol_idx for (_, mol_idx) in top50]

    out = []
    for mol_idx in conf_ids:
        conf = mol.GetConformer(mol_idx)
        id = conf.GetId()
        new_mol = Chem.Mol(mol, confId=id)
        e = next(e for (e, idx2) in pairs if idx2 == mol_idx)
        g = mol_to_e3fp(new_mol, conf, bits, level, radius, group_id=idx, conf_energy=e)
        if g is None:
            print("skipped conformer")
        else:
            g.y = torch.tensor([perm], dtype=torch.float)
            out.append(g)
    print(len(out))
    return out

class ConformerDataset(InMemoryDataset):
    """
    A tiny InMemoryDataset that reads/writes exactly the single .pt file
    you pass to it.
    """
    def __init__(self, processed_path, data_list=None):
        self._processed_path = processed_path
        self._processed_file = os.path.basename(processed_path)
        root = os.path.dirname(processed_path)
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

        if data_list is not None:
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self._processed_path)
        else:
            self.data, self.slices = torch.load(self._processed_path)
    def set_bits(self, bits):
        self.bits = bits
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self._processed_file]

    def download(self):
        pass

    def get(self, idx):
        # this pulls out the flattened slices for example `idx`
        data = super().get(idx)

        # how many conformers?
        n_conf = data.mol_id.numel()   # or len(data.mol_id)

        # reshape the flat fingerprint back to [n_conf, 2048]
        data.e3fp = data.e3fp.view(n_conf, self.bits)

        return data

def preprocess(raw_dir, pickle_dir, processed_dir, num_workers, log_dir, bits, level, radius):
    os.makedirs(processed_dir, exist_ok=True)
    
    for split in ["raw_train.csv", "raw_val.csv", "raw_test.csv"]:
        out_file = os.path.join(
            processed_dir,
            split.replace("raw_", "e3fp_").replace(".csv", ".pt")
        )
        if os.path.exists(out_file):
            logging.info(f"{out_file} exists, skip.")
            continue

        rows = pd.read_csv(os.path.join(raw_dir, split)).to_dict("records")
        logging.info(f"Processing {split}   (#rows={len(rows)})")

        t0 = time.time()
        with Pool(num_workers) as pool:
            nested = list(tqdm(
                pool.imap(process_single_row,  [(i, r, pickle_dir, bits, level, radius) for i, r in enumerate(rows)]),
                total=len(rows),
                desc=f"→ {split}"
            ))
            
        #print(len(nested))
        #print(len(nested[0]))
        unpacked = []
        for mol in nested:
            for conf in mol:
                unpacked.append(conf)
        #print(len(unpacked)) this was 50
        #print(unpacked[1]) this was output of process single row for 1 conformer
       # flat      = [g for seq in nested for g in seq]
        #data_objs = [d for d in nested if d is not None]
       # print(len(data_objs))  # e.g. 10,000 conformers                                                   
      #  print(data_objs[0].e3fp.shape)  # should be torch.Size([2048])
        ds = ConformerDataset(out_file, unpacked)
        ds.set_bits(bits)

        logging.info(f"Saved {len(unpacked)} conformers to {out_file} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--pickle_dir", required=True)
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--bits", type=int, default=2048)
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--radius", type=float, default=1.5)
    args = p.parse_args()
    preprocess(**vars(args))
    
