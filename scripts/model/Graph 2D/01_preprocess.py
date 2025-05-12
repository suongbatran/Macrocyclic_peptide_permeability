import os, pickle
import pandas as pd
import argparse
from pathos.multiprocessing import ProcessingPool as Pool
import logging
from datetime import datetime
import time
import torch
from tqdm import tqdm

from utils import load_smiles, smiles_to_graph, seed_everything
from rdkit import RDLogger
from torch_geometric.data import Data
RDLogger.DisableLog('rdApp.*')

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = os.path.join(log_dir, f"preprocess_log_{current_time}.txt")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def process_single_row(args):
    '''Preprocess each peptide'''
    row, pickle_dir = args
    seq = row["sequence"]
    permeability = row["permeability"]

    pickle_path = os.path.join(pickle_dir, f"{seq}.pickle")
    if not os.path.exists(pickle_path):
        logging.error(f"Pickle file not found for sequence: {seq} in {pickle_dir}")
        return None
    smiles = load_smiles(pickle_path)
    
    x, _, e_attr, e_pairs = smiles_to_graph(smiles)
    edge_index = torch.tensor(e_pairs, dtype=torch.long).t().contiguous()

    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=torch.tensor(e_attr, dtype=torch.float),
                y=torch.tensor([permeability], dtype=torch.float))
    return data

def preprocess(raw_dir, pickle_dir, processed_dir, num_workers, log_dir):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
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
        with Pool(num_workers) as pool:
            data_objs = list(tqdm(pool.imap(process_single_row, [(r, pickle_dir) for r in rows]),
                                  total=len(rows),
                                  desc=f"→ {split}"))

        data_objs = [d for d in data_objs if d is not None]
        torch.save(data_objs, out_file)
        logging.info(f"Saved {len(data_objs)} graphs to {out_file} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",        required=True)
    p.add_argument("--pickle_dir",     required=True)
    p.add_argument("--processed_dir",  required=True)
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--log_dir")
    preprocess(**vars(p.parse_args()))
