"""
Lightning 2.5.1 implementation with optional grid-search.
Takes 50 lowest energy conformers for each molecule, processes graphs, then pools representations 
before making final prediction
"""
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
warnings.filterwarnings(
    "ignore",
    message="The `srun` command is available",
    module="lightning_fabric.plugins.environments.slurm"
)

import argparse, os, itertools, json, torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from utils import seed_everything
from torch_scatter import scatter_add, scatter_mean, scatter_softmax
from torch_geometric.nn.models import SchNet as SchNetModel   
from pytorch_lightning.strategies import DDPStrategy
import random
from collections import defaultdict
from torch.utils.data import Sampler
from pytorch_lightning.loggers import TensorBoardLogger
from statistics import mean
import time
import math
import copy

from preprocess import ConformerDataset
torch.backends.cudnn.benchmark = True

def edge_mlp(e_in, hidden):
    return Sequential(Linear(e_in, hidden), ReLU(),
                      Linear(hidden, hidden), ReLU(),
                      Linear(hidden, hidden))

class SchNetLit(pl.LightningModule):
    def __init__(self, hidden=128, lr=1e-4, agg="mean", num_interactions=3, cutoff=10):
        super().__init__()
        self.save_hyperparameters()
        self.schnet = SchNetModel(hidden_channels=hidden, num_filters=hidden,
                                  num_interactions=num_interactions, cutoff=cutoff)
        self.lin = torch.nn.Linear(hidden, 1)
        self.validation_step_losses = []

    def forward(self, data):
        # Your forward method remains the same
        node_to_conf = data.batch  
        h = self.schnet.embedding(data.z)  
        edge_index, edge_weight = self.schnet.interaction_graph(data.pos, node_to_conf)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        conf_reps = scatter_mean(h, node_to_conf, dim=0)  
        return conf_reps

    def training_step(self, batch, batch_idx):
        # Get model representations
        conf_reps = self(batch)                             
        conf_pred = self.lin(conf_reps).view(-1)            

        # Pool conformers to get molecule-level predictions
        conf_idx, mol_ids = batch.ptr[:-1], batch.mol_id[batch.ptr[:-1]]
        unique_ids, inv = torch.unique(mol_ids, sorted=True, return_inverse=True)
        mol_pred = pool_conformers(conf_pred, inv, self.hparams.agg)  
        y_true = pool_conformers(batch.y, inv, "mean")        

        # Calculate loss
        loss = F.mse_loss(mol_pred, y_true)
        
        # Log training metrics
        self.log("train_mse", loss, prog_bar=True, on_step=True, on_epoch=True, 
                batch_size=y_true.size(0), sync_dist=True)
        
        return loss

    def on_train_epoch_end(self):
        print(f"End of training epoch {self.current_epoch}, about to run validation")
    
    def validation_step(self, batch, batch_idx):
        # Do calculations separately for validation
        print((f"Running validation_step {batch_idx}"))

        conf_reps = self(batch)                             
        conf_pred = self.lin(conf_reps).view(-1)            
        
        conf_idx, mol_ids = batch.ptr[:-1], batch.mol_id[batch.ptr[:-1]]
        unique_ids, inv = torch.unique(mol_ids, sorted=True, return_inverse=True)
        mol_pred = pool_conformers(conf_pred, inv, self.hparams.agg)  
        y_true = pool_conformers(batch.y, inv, "mean")        
        
        # Calculate loss
        val_loss = F.mse_loss(mol_pred, y_true)
        
        # Log individual steps but NOT on_epoch
        print(f"Attempting to append loss {val_loss}")
        try:
            self.validation_step_losses.append(val_loss)
            print("Successfully appended to validation_step_losses")
        except AttributeError as e:
            print(f"ERROR: {e}")

        self.log("val_mse_step", val_loss, prog_bar=True, on_step=True, on_epoch=False, 
        batch_size=y_true.size(0), sync_dist=True)
                    
        return val_loss
    
    def on_validation_epoch_start(self):
        # Reset validation losses
        self.validation_step_losses = []

    def on_validation_epoch_end(self):
        # Proper handling of validation metrics in distributed environment

        print("Running validation_epoch_end")
        print(f"validation_step_losses exists: {hasattr(self, 'validation_step_losses')}")
        if hasattr(self, 'validation_step_losses'):
            print(f"len(validation_step_losses): {len(self.validation_step_losses)}")

        if self.validation_step_losses:
            # First, convert list to tensor
            val_losses = torch.stack(self.validation_step_losses)
            
            # Gather losses from all processes if in distributed mode
            if torch.distributed.is_initialized():
                # Create a tensor to hold gathered losses
                world_size = torch.distributed.get_world_size()
                gathered_losses = [torch.zeros_like(val_losses) for _ in range(world_size)]
                
                # Gather losses from all processes
                torch.distributed.all_gather(gathered_losses, val_losses)
                
                # Concatenate all gathered losses
                all_losses = torch.cat(gathered_losses)
                
                # Calculate mean across all processes
                val_loss_mean = all_losses.mean()
            else:
                # In non-distributed mode, just take the mean of local losses
                val_loss_mean = val_losses.mean()
            
            # Store the loss
            self.val_loss_mean = val_loss_mean
            
            # Log the epoch-level metric - CRITICAL for checkpoint callback
            self.log("val_mse", val_loss_mean, prog_bar=True, sync_dist=True)

        print("Finished validation_epoch_end")
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'callback_metrics'):
            print(f"callback_metrics contains: {list(self.trainer.callback_metrics.keys())}")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
    
class GroupSampler(Sampler[list]):
    """Yield batches of `groups_per_batch` molecules with DDP compatibility."""
    def __init__(self, dataset, groups_per_batch=10,
                 shuffle=True, drop_last=False,
                 num_replicas=1, rank=0):
        self.gpb = groups_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas  # Total number of processes
        self.rank = rank  # Current process rank
        self.epoch = 0
        
        buckets, sizes = defaultdict(list), []
        # Store molecule ID to indices mapping
        for idx, data in enumerate(dataset):
            # Get molecule ID safely
            mol_id = int(data.mol_id.view(-1)[0])
            buckets[mol_id].append(idx)

        # Convert buckets to a list for better pickling
        self.groups = list(buckets.values())
        sizes = [len(g) for g in self.groups]

        # stats - needed for logging
        self.n_groups = len(self.groups)
        self.n_short = sum(1 for s in sizes if s < 50)
        self.short_rate = self.n_short / self.n_groups
        self.avg_conf_per_mol = mean(sizes)
        
        # For DDP: Partition groups to each process
        if self.num_replicas > 1:
            # Ensure even distribution across processes
            self.num_samples = math.ceil(len(self.groups) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            
            # Add extra samples to make it evenly divisible
            if len(self.groups) % self.num_replicas != 0:
                # Pad with repeated indices to make it evenly divisible
                padding_size = self.total_size - len(self.groups)
                self.groups.extend(self.groups[:padding_size])
            
            # Subsample for this rank
            assert len(self.groups) == self.total_size
            offset = self.num_samples * self.rank
            self.groups = self.groups[offset:offset + self.num_samples]
            assert len(self.groups) == self.num_samples
        else:
            self.num_samples = len(self.groups)
            self.total_size = len(self.groups)
    
    def set_epoch(self, epoch):
        # Set epoch for reproducible shuffling across epochs
        self.epoch = epoch
        
    def __iter__(self):
        if self.shuffle:
            # Use the epoch in the random seed for deterministic shuffling
            g = torch.Generator()
            g.manual_seed(self.epoch)
            order = torch.randperm(len(self.groups), generator=g).tolist()
        else:
            order = list(range(len(self.groups)))
            
        bucket = []
        for idx in order:
            gid = idx
            bucket.extend(self.groups[gid])
            if len(bucket) >= self.gpb * 50:         
                yield bucket
                bucket = []
        if bucket and not self.drop_last:
            yield bucket

    def __len__(self):
        full = self.n_groups // self.gpb
        return full if (self.n_groups % self.gpb == 0 or self.drop_last) else full + 1

class SamplerStats(pl.Callback):
    def __init__(self, sampler, name="train"):
        self.sampler = sampler
        self.name = name

    def on_train_start(self, trainer, pl_module):
        # Only print once in distributed mode
        if trainer.global_rank == 0:
            s = self.sampler
            print(f"[{self.name}] short‑mol: {s.n_short}/{s.n_groups} "
                f"({s.short_rate:.2%})  avg_confs={s.avg_conf_per_mol:.2f}")

    def on_train_epoch_end(self, trainer, pl_module):
        s = self.sampler
        pl_module.log_dict({
            f"{self.name}_short_rate": s.short_rate,
            f"{self.name}_avg_confs": s.avg_conf_per_mol,
            f"{self.name}_n_short": float(s.n_short),
        }, prog_bar=True, sync_dist=True)  # Added sync_dist=True

class SamplerEpochReset(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        bs = trainer.train_dataloader.batch_sampler
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(trainer.current_epoch)

def make_loaders(processed_dir, num_workers, groups_per_batch, world_size=1, rank=0):
    train_ds = ConformerDataset(os.path.join(processed_dir, "train.pt"))
    val_ds = ConformerDataset(os.path.join(processed_dir, "val.pt"))

    # Use DDP-aware samplers
    gsam_train = GroupSampler(train_ds, groups_per_batch=groups_per_batch, 
                             shuffle=True, num_replicas=world_size, rank=rank)
    gsam_val = GroupSampler(val_ds, groups_per_batch=groups_per_batch, 
                           shuffle=False, num_replicas=world_size, rank=rank)

    # For persistent_workers=True, we need to ensure num_workers >= 1
    effective_workers = max(1, num_workers) if num_workers > 0 else 0
    
    # Only use persistent workers if we're using workers
    persistent = num_workers > 0 and effective_workers > 0

    train_dl = DataLoader(
        train_ds, 
        batch_sampler=gsam_train,
        num_workers=effective_workers, 
        persistent_workers=persistent,
        pin_memory=torch.cuda.is_available()
    )
    
    val_dl = DataLoader(
        val_ds,   
        batch_sampler=gsam_val,
        num_workers=effective_workers, 
        persistent_workers=persistent,
        pin_memory=torch.cuda.is_available()
    )

    return train_dl, val_dl, gsam_train, gsam_val

def pool_conformers(pred, mol_id, method):
    if method == "sum":
        return scatter_add(pred, mol_id, dim=0)
    if method == "mean":
        return scatter_mean(pred, mol_id, dim=0)
    if method == "softmax":
        w = scatter_softmax(pred, mol_id, dim=0)
        return scatter_add(w * pred, mol_id, dim=0)
    if method == "softmin":
        w = scatter_softmax(-pred, mol_id, dim=0)
        return scatter_add(w * pred, mol_id, dim=0)
    raise ValueError(f"Unknown agg {method}")

def gridsearch(args):
    # Map each hyperparam name to the constructor you want applied to each token
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = args.devices
    type_map = {
        "hidden":               int,
        "num_interactions":     int,
        "lr":                   float,
        "agg":                  str,
        "cutoff":               float,
    }

    # Build a dict of lists
    grid = {}
    for k, ctor in type_map.items():
        raw = getattr(args, k)
        # allow comma-separated values in a single arg
        if isinstance(raw, str) and "," in raw:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            parts = [raw]

        # cast each part
        try:
            grid[k] = [ctor(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"Could not parse {k} values {parts!r} with {ctor}: {e}")

    # Print info only on the main process to avoid duplication
    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
    
    # Now build every combination
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    
    if is_main_process:
        print(f"▶ sweeping {len(combos)} hyper-parameter sets")
    
    best_mse, best_cfg = 1e9, None
    times = []    # store elapsed times per combo
    
    for combo_id, values in enumerate(combos, 1):
        cfg = dict(zip(grid.keys(), values))
        run_name = "_".join(f"{k}{v}" for k,v in cfg.items())
        ckpt_dir = args.checkpoint_dir; os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n>>> Starting combo {combo_id}/{len(combos)}: {cfg}")

        # data loaders
        train_dl, val_dl, gsam_train, gsam_val = make_loaders(
            args.processed_dir, args.num_workers, args.groups_per_batch, 
            world_size=world_size, rank=rank
        )

        # model & trainer
        model = SchNetLit(hidden=cfg["hidden"], lr=cfg["lr"], agg=cfg["agg"], 
                          num_interactions=cfg["num_interactions"], cutoff=cfg["cutoff"])
        
        es_cb = EarlyStopping(
            monitor="val_mse",
            mode="min",
            patience=args.patience,
            verbose=is_main_process,  # Only verbose on main process
            check_on_train_epoch_end=False,
        )
        
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{val_mse:.4f}",
            monitor="val_mse",
            mode="min",
            save_top_k=1,
            verbose=is_main_process
        )
        sampler_cb = SamplerStats(gsam_train, name="train")
        sampler_reset_cb = SamplerEpochReset()
        logger = TensorBoardLogger(save_dir=args.log_dir, name=run_name)

        num_batches    = len(train_dl)   

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.devices,
            strategy=DDPStrategy(
                find_unused_parameters=True, # Must be True for this model type
                static_graph=False,
            ),
            max_epochs=args.epochs,
            precision="16-mixed",
            callbacks=[es_cb, ckpt_cb, sampler_cb, sampler_reset_cb],
            logger=logger,  # Use a proper logger
            enable_progress_bar=is_main_process,
            enable_model_summary=is_main_process,
            check_val_every_n_epoch=None,           # disable epoch-based scheduling
            val_check_interval= num_batches // args.devices,   # integer → run after N train batches
            limit_val_batches=1.0,                  # still use 100% of val set
            num_sanity_val_steps=2,
            use_distributed_sampler=False)

        # ── timing starts ─────────────────────────
        t0 = time.time()
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        elapsed = time.time() - t0
        times.append(elapsed)
        avg_time = sum(times) / len(times)
        # ── timing ends ───────────────────────────

        # Only print on rank 0 to avoid duplication
        print(f"<<< Finished training {combo_id} in {elapsed:.1f}s, appending val mse now")
        if is_main_process:
            if ckpt_cb.best_model_score is not None:
                val_mse = ckpt_cb.best_model_score.item()
            else:
                val_tensor = trainer.callback_metrics.get("val_mse")
                if val_tensor is None:
                    raise RuntimeError("No val_mse found! Did you log it in on_validation_epoch_end?")
                val_mse = val_tensor.item()
                
            print(f"[{combo_id}/{len(combos)}] {run_name}  "
                f"valMSE={val_mse:.4f}  "
                f"time={elapsed:.1f}s  avg_time={avg_time:.1f}s")

    # Only print on rank 0
    if is_main_process and best_cfg is not None:
        print(f"◆ best config: {json.dumps(best_cfg)}   valMSE={best_mse:.4f}")

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--log_dir", required=True)         
    p.add_argument("--hidden",     default="64")        # CLI accepts comma-lists
    p.add_argument("--num_interactions",     default="3")
    p.add_argument("--lr",         default="1e-3")
    p.add_argument("--agg",        default="mean")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--patience",   type=int, default=20)
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--devices",    type=int, default=1)
    p.add_argument("--cutoff",     default="10.0")
    p.add_argument("--groups_per_batch",    type=int, default=10)
    args = p.parse_args()
    valid = {"sum","mean","softmax","softmin"}
    for part in args.agg.split(","):
        if part not in valid:
            p.error(f"Invalid --agg value: {part!r}")
    args.num_workers = max(1, min(args.num_workers, 16))
    # create dirs
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    gridsearch(args)