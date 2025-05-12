"""
Load test.pt + best_gnn.pt → save predictions & metrics
"""
import argparse, csv, logging, os, time, torch, pytorch_lightning as pl
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.loader import DataLoader
from train import GNN2DLit      

def log_setup(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(filename=os.path.join(log_dir, f"predict_{ts}.log"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

class GNN2DLitPred(GNN2DLit):
    def predict_step(self, batch, batch_idx):
        y_hat = self(batch)
        seqs  = getattr(batch, "sequence", [""] * batch.num_graphs)
        return {"pred": y_hat.cpu(), "true": batch.y.cpu(), "seq": seqs}

def predict(processed_dir, checkpoint_dir, result_dir, log_dir):
    log_setup(log_dir)
    os.makedirs(result_dir, exist_ok=True)

    test_ds = torch.load(os.path.join(processed_dir, "test.pt"))
    loader  = DataLoader(test_ds, batch_size=64)

    ckpt_path = os.path.join(checkpoint_dir, "best_gnn.pt")
    model = GNN2DLitPred.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer(accelerator="gpu", devices=[0],
                         logger=False, enable_progress_bar=True)

    t0 = time.time()
    out_batches = trainer.predict(model, loader)          
    elapsed = time.time() - t0

    preds = torch.cat([b["pred"] for b in out_batches]).numpy()
    ys    = torch.cat([b["true"] for b in out_batches]).numpy()
    seqs  = [s for b in out_batches for s in b["seq"]]

    csv_path = os.path.join(result_dir, "test_predict.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "predict_permeability"])
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
    predict(**vars(p.parse_args()))