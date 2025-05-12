from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import torch
import pickle
import heapq, argparse
import os, csv, logging, time
from datetime import datetime
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from utils import mol_to_graph, seed_everything
from multiprocessing import get_context
import torch.multiprocessing as mp
from torch_geometric.data import InMemoryDataset, Data, Batch
RDLogger.DisableLog('rdApp.*')
mp.set_sharing_strategy('file_system')

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = os.path.join(log_dir, f"preprocessing_log_{current_time}.txt")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def process_single_row(args):
    """
    returns *list* of Data objects (one per conformer) so that Pool-imap
    keeps the original API.
    """
    idx, row, pickle_dir = args
    seq, perm = row["sequence"], row["permeability"]

    pkl = os.path.join(pickle_dir, f"{seq}.pickle")
    if not os.path.exists(pkl):
        logging.error(f"Pickle missing for {seq}")
        return []

    with open(pkl, "rb") as fh:
        d = pickle.load(fh)

        mol = d["rd_mol"]
    conf_dicts = d.get("conformers", [])
    if not conf_dicts:
        logging.warning(f"No conformer metadata for {seq}")
        return []

    # build (energy, mol_conformer_idx) pairs
    # RDKit convention: 'set' is 1-based, so subtract 1
    pairs = []
    for conf in conf_dicts:
        set_id = conf.get("set")
        if set_id is None:
            continue
        mol_idx = set_id - 1
        # guard against out‐of‐bounds
        if not (0 <= mol_idx < mol.GetNumConformers()):
            logging.warning(f"Bad set index {set_id} for {seq}, skipping")
            continue
        energy = conf.get("relativeenergy", None)
        if energy is None:
            logging.warning(f"Missing relativeenergy for conformer {set_id} of {seq}")
            continue
        pairs.append((energy, mol_idx))

    # pick the 50 lowest‐energy ones
    top50 = heapq.nsmallest(50, pairs, key=lambda x: x[0])
    conf_ids = [mol_idx for (_, mol_idx) in top50]

    out = []
    for mol_idx in conf_ids:
        conf = mol.GetConformer(mol_idx)
        # find the energy again (optional, or you could store in a dict above)
        e = next(e for (e, idx2) in pairs if idx2 == mol_idx)
        g = mol_to_graph(mol, conf, group_id=idx, conf_energy=e)
        g.y = torch.tensor([perm], dtype=torch.float)
        out.append(g)

    return out

class ConformerDataset(InMemoryDataset):
    """
    A tiny InMemoryDataset that reads/writes exactly the single .pt file
    you pass to it.
    """
    def __init__(self, processed_path, data_list=None):
        # store full path and filename
        self._processed_path = processed_path
        self._processed_file = os.path.basename(processed_path)
        root = os.path.dirname(processed_path)
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

        if data_list is not None:
            # first time: collate and save
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self._processed_path)
        else:
            # later: load from disk
            self.data, self.slices = torch.load(self._processed_path)

    @property
    def raw_file_names(self):
        return []  # not used

    @property
    def processed_file_names(self):
        # must match exactly the basename of processed_path
        return [self._processed_file]

    def download(self):
        pass

def preprocess(raw_dir, pickle_dir, processed_dir, num_workers, log_dir):
    os.makedirs(processed_dir, exist_ok=True)
    setup_logging(log_dir)
    seed_everything()

    for split in ["raw_train.csv", "raw_val.csv", "raw_test.csv"]:
        out_file = os.path.join(processed_dir, split.replace("raw_", "").replace(".csv", ".pt"))
        if os.path.exists(out_file):
            logging.info(f"{out_file} exists, skip.")
            continue

        rows = pd.read_csv(os.path.join(raw_dir, split)).to_dict("records")
        logging.info(f"Processing {split}   (#rows={len(rows)})")

        t0 = time.time()
        with ThreadPool(num_workers) as pool:
            # pool returns list[list[Data]] ; flatten once
            tmp = pool.imap(process_single_row, [(i, r, pickle_dir) for i,r in enumerate(rows)])
            data_objs = list(tqdm(tmp, total=len(rows), desc=f"→ {split}"))
            data_objs = [d for sub in data_objs for d in sub]

        data_objs = [g for g in data_objs if g]
        ds = ConformerDataset(out_file, data_objs)
        logging.info(f"Saved {len(data_objs)} graphs to {out_file} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",        required=True)
    p.add_argument("--pickle_dir",     required=True)
    p.add_argument("--processed_dir",  required=True)
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--log_dir",        required=True)
    preprocess(**vars(p.parse_args()))
