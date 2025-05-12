"""
Load test.pt + best_gnn.pt → save predictions & metrics
"""
import argparse, csv, logging, os, time, torch, pytorch_lightning as pl
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.loader import DataLoader
from train import SchNetLit
from torch_scatter import scatter_mean     
from train import pool_conformers 
from preprocess import ConformerDataset
import time

def log_setup(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(filename=os.path.join(log_dir, f"predict_{ts}.log"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

class SchNetLitPred(SchNetLit):
    def predict_step(self, batch, batch_idx):
        # batch is now a PyG DataBatch with .batch, .ptr, .mol_id, .y
        conf_reps = self(batch)                  # same SchNet forward
        y_hat     = self.lin(conf_reps).view(-1) # [n_confs]

        # recover molecule-ids per conformer
        conf_idx = batch.ptr[:-1]
        mol_ids  = batch.mol_id[conf_idx]        # [n_confs]

        seqs = getattr(batch, "sequence", [""] * y_hat.size(0))
        return {
          "pred":   y_hat.cpu(),
          "true":   batch.y.cpu(),
          "mol_id": mol_ids.cpu(),
          "seq":    seqs,
        }

def predict(processed_dir, checkpoint_dir, result_dir, log_dir, agg):
    log_setup(log_dir)
    os.makedirs(result_dir, exist_ok=True)

    test_ds = ConformerDataset(os.path.join(processed_dir, "test.pt"))
    loader = DataLoader(test_ds, batch_size=64, pin_memory=True)

    ckpt_path = os.path.join(checkpoint_dir, "best_gnn.pt")
    model = SchNetLitPred.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer(accelerator="auto", devices="auto",
                         logger=False, enable_progress_bar=True)

    t0 = time.time()
    out_batches = trainer.predict(model, loader)          
    elapsed = time.time() - t0

    all_seqs = sum((b["seq"] for b in out_batches), [])
    all_mol_ids = torch.cat([b["mol_id"] for b in out_batches]).numpy()

    # collect everything, now including mol_id
    conformer_preds = torch.cat([b["pred"]   for b in out_batches])
    mol_ids         = torch.cat([b["mol_id"] for b in out_batches])
    ys              = torch.cat([b["true"]   for b in out_batches])

    # pool to molecule-level
    preds = pool_conformers(conformer_preds, mol_ids, agg).numpy()
    ys    = scatter_mean(ys, mol_ids, dim=0).numpy()

    seq_map = {}
    for mid, seq in zip(all_mol_ids, all_seqs):
        if mid not in seq_map:
            seq_map[mid] = seq

    # in the same order as preds/ys
    unique_ids, inv = torch.unique(torch.tensor(all_mol_ids), return_inverse=True)
    mol_order = unique_ids.numpy()  # these are the distinct molecule IDs in batch order
    seqs = [seq_map[mid] for mid in mol_order]

    csv_path = os.path.join(result_dir, "test_predict.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(seqs, preds))
    logging.info(f"Saved predictions → {csv_path}")

    r2, mse = r2_score(ys, preds), mean_squared_error(ys, preds)
    msg = f"R²={r2:.4f}   MSE={mse:.4f}   time={elapsed:.1f}s"
    logging.info(msg)
    print(msg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir",  required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--result_dir",     required=True)
    p.add_argument("--log_dir",        required=True)
    p.add_argument("--agg",        choices=["sum","mean","softmax","softmin"])
    predict(**vars(p.parse_args()))