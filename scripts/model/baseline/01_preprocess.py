
import os
import pandas as pd
import argparse
from pathos.multiprocessing import ProcessingPool as Pool
import csv
import logging
from datetime import datetime
import time
from utils import load_smiles, compute_tpsa_from_smiles

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
    '''Preprocess each peptide'''
    row, pickle_dir = args
    seq = row["sequence"]
    permeability = row["permeability"]

    pickle_path = os.path.join(pickle_dir, f"{seq}.pickle")
    if not os.path.exists(pickle_path):
        logging.error(f"Pickle file not found for sequence: {seq} in {pickle_dir}")
        print(f"Pickle file not found for sequence: {seq} in {pickle_dir}")
        return None
    smiles = load_smiles(pickle_path)
    tpsa = compute_tpsa_from_smiles(smiles)
    return {
        "sequence": seq,
        "permeability": permeability,
        "2D_TPSA": tpsa,
    }

def preprocess(raw_dir, pickle_dir, processed_dir, num_workers, log_dir):
    '''To preprocess all 3 train, test, val files '''
    os.makedirs(processed_dir, exist_ok=True)

    # Set up logging
    setup_logging(log_dir)

    start_total = time.time()

    for split in ["raw_train.csv", "raw_test.csv", "raw_val.csv"]:
        if split.replace("raw_", "") in os.listdir(processed_dir):
            logging.info(f"Found processed file {split.replace('raw_', '')} in {processed_dir}, skipping!")
            continue
        raw_path = os.path.join(raw_dir, split)
        processed_path = os.path.join(processed_dir, split.replace("raw_", ""))

        start_split = time.time()
        logging.info(f"Start processing {split}...")

        # Saving processed result
        header = ["sequence", "2D_TPSA", "permeability"]
        with open(processed_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

        df_iter = pd.read_csv(raw_path, chunksize=1)

        with Pool(num_workers) as pool:
            for batch in df_iter:
                row = batch.iloc[0].to_dict()
                result = pool.map(process_single_row, [(row, pickle_dir)])[0]

                if result is not None:
                    with open(processed_path, mode='a', newline='') as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=header)
                        writer.writerow(result)

        end_split = time.time()
        logging.info(f"Finished processing {split} in {end_split - start_split:.2f} seconds.")
        logging.info(f"Saved processed file to {processed_path}")

    end_total = time.time()
    logging.info(f"Total preprocessing time: {end_total - start_total:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str)
    parser.add_argument("--pickle_dir", type=str)
    parser.add_argument("--processed_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str) 
    args = parser.parse_args()

    preprocess(args.raw_dir, args.pickle_dir, args.processed_dir, args.num_workers, args.log_dir)