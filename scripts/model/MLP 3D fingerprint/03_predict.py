import argparse
import os
import random
import csv
import time
from datetime import datetime
import logging
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from train import E3FPRegressor, pool_conformers
from preprocess import ConformerDataset


def log_setup(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(filename=os.path.join(log_dir, f"predict_{ts}.log"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def predict(processed_dir, checkpoint_dir, result_dir, log_dir, bits, hidden, agg, device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    test_ds = ConformerDataset(os.path.join(processed_dir, "e3fp_test.pt"))
    test_ds.set_bits(bits)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = E3FPRegressor(bits=bits, hidden=hidden).to(device)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    log_setup(log_dir)

    start = time.time()
    all_conf_preds = []
    all_mol_ids = []
    all_true = []
    all_seqs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            conf_reps = model.forward_conf(batch)
            conf_pred = model.lin(conf_reps).view(-1)

            all_conf_preds.append(conf_pred.cpu())
            mol_ids = batch.mol_id
            all_mol_ids.append(mol_ids.cpu())

            all_true.append(batch.y.cpu())
            seqs = getattr(batch, 'sequence', None)
            if seqs is not None:
                all_seqs.extend(seqs)

    elapsed = time.time() - start

    conf_preds = torch.cat(all_conf_preds)
    mol_ids = torch.cat(all_mol_ids)
    true_confs = torch.cat(all_true)

    preds = pool_conformers(conf_preds, mol_ids, agg).numpy()
    ys = scatter_mean(true_confs, mol_ids, dim=0).numpy()

    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, 'test_predict.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['mol_id', 'predict_permeability']
        if all_seqs:
            header.insert(1, 'sequence')
        writer.writerow(header)
        for idx, val in enumerate(preds):
            row = [idx]
            if all_seqs:
                row.append(all_seqs[idx])
            row.append(val)
            writer.writerow(row)

    r2 = r2_score(ys, preds)
    mse = mean_squared_error(ys, preds)
    msg = f"R²={r2:.4f}   MSE={mse:.4f}   time={elapsed:.1f}s"
    logging.info(msg + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--bits', type=int, default=2048)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--agg', choices=['sum','mean','softmax','softmin'], default='mean')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    predict(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        result_dir=args.result_dir,
        log_dir=args.log_dir,
        bits=args.bits,
        hidden=args.hidden,
        agg=args.agg,
        device=args.device
    )