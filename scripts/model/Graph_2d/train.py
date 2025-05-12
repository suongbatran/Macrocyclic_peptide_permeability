"""
Lightning 2.5.1 implementation with optional grid-search.

EXAMPLE
-------
# single run (like before)
python train_gnn_pl.py --processed_dir torch_data --checkpoint_dir ckpt --log_dir logs \
                       --hidden 64 --layers 3 --batch_size 32 --lr 1e-3

# grid-search over 3×2×2 = 12 configs
python train_gnn_pl.py --processed_dir torch_data --checkpoint_dir ckpt --log_dir logs \
                       --hidden 64,128,256 --layers 3,5 --lr 1e-3,5e-4
"""
import argparse, os, logging, itertools, json, torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from utils import seed_everything   
from datetime import datetime

import warnings
warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'pyg-lib'",
    module="torch_geometric.typing"
)
warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'torch-sparse'",
    module="torch_geometric.typing"
)

def log_setup(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(filename=os.path.join(log_dir, f"train_{ts}.log"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def gin_mlp(hidden: int):
    return Sequential(
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, hidden)
    )

class GNN2DLit(pl.LightningModule):
    def __init__(self, node_dim, edge_dim, hidden=64, layers=3, lr=1e-3):
        super().__init__()
        layers = int(layers)
        self.save_hyperparameters()          # logs hyperparams to checkpoint
        self.node_lin = torch.nn.Linear(node_dim, hidden)
        self.convs = torch.nn.ModuleList([
            GINEConv(
                nn=gin_mlp(hidden),      # maps hidden→hidden
                train_eps=True,
                edge_dim=edge_dim        # so GINEConv builds edge_encoder: 6→64
            )
            for _ in range(layers)
        ])
        self.bns   = torch.nn.ModuleList([BatchNorm1d(hidden) for _ in range(layers)])
        self.lin1, self.lin2 = Linear(hidden, hidden), Linear(hidden, 1)

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_lin(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei, ea)))
        x = global_mean_pool(x, batch)
        return self.lin2(F.relu(self.lin1(x))).view(-1)

    def _shared_step(self, batch, stage):
        pred = self(batch)
        y    = batch.y
        mse  = F.mse_loss(pred, y)
        self.log(f"{stage}_mse", mse, prog_bar=True, on_epoch=True, batch_size=y.size(0), sync_dist=True)
        return mse

    def training_step(self, batch, _):   return self._shared_step(batch, "train")
    def validation_step(self, batch, _): return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

def make_loaders(processed_dir, batch_size):
    train_ds = torch.load(os.path.join(processed_dir, "train.pt"))
    val_ds   = torch.load(os.path.join(processed_dir, "val.pt"))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, num_workers=4, persistent_workers=True)
    node_dim = train_ds[0].x.size(1)
    edge_dim = train_ds[0].edge_attr.size(1)
    return train_dl, val_dl, node_dim, edge_dim

def gridsearch(args):
    # build grid dict: key -> list of values
    grid = {}
    for k in ["hidden", "layers", "batch_size", "lr"]:
        raw = getattr(args, k)
        if isinstance(raw, str) and "," in raw:
            parts = raw.split(",")
        else:
            parts = [raw]

        # now cast each part to the right type
        if k in ("hidden", "layers", "batch_size"):
            # integer hyperparams
            grid[k] = [int(p) for p in parts]
        elif k == "lr":
            # learning‐rate is float
            grid[k] = [float(p) for p in parts]
        else:
            grid[k] = parts

    combos = list(itertools.product(*grid.values()))
    logging.info(f"▶ sweeping {len(combos)} hyper-parameter sets")
    best_mse, best_cfg = 1e9, None

    for combo_id, values in enumerate(combos, 1):
        cfg = dict(zip(grid.keys(), values))
        run_name = "_".join(f"{k}{v}" for k,v in cfg.items())
        ckpt_dir = os.path.join(args.checkpoint_dir, run_name); os.makedirs(ckpt_dir, exist_ok=True)

        # data loaders
        train_dl, val_dl, node_dim, edge_dim = make_loaders(args.processed_dir, cfg["batch_size"])

        # model & trainer
        cfg.pop("batch_size")
        model = GNN2DLit(node_dim, edge_dim, **cfg)
        es_cb = EarlyStopping(monitor="val_mse", mode="min", patience=args.patience, verbose=False)
        ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir,
                                  filename="best",
                                  monitor="val_mse",
                                  mode="min",
                                  save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[es_cb, ckpt_cb],
            logger=False,
            accelerator="gpu",
            devices=[0],
            enable_progress_bar=True
        )
        seed_everything()            
        trainer.fit(model, train_dl, val_dl)   # training

        val_mse = ckpt_cb.best_model_score.item()
        logging.info(f"[{combo_id}/{len(combos)}] {run_name}  valMSE={val_mse:.4f}")
        if val_mse < best_mse:
            best_mse, best_cfg = val_mse, cfg
            # keep a copy at the *root* checkpoint_dir
            root_best = os.path.join(args.checkpoint_dir, "best_gnn.pt")
            trainer.save_checkpoint(root_best)

    logging.info(f"◆ best config: {json.dumps(best_cfg)}   valMSE={best_mse:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--log_dir", required=True)         
    p.add_argument("--hidden",     default="64")        # CLI accepts comma-lists
    p.add_argument("--layers",     default="3")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         default="1e-3")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--patience",   type=int, default=20)
    args = p.parse_args()

    # create dirs
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_setup(args.log_dir)

    gridsearch(args)