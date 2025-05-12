import argparse
import os
import random
from collections import defaultdict
from statistics import mean
from datetime import datetime
import time
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

from preprocess import ConformerDataset

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = os.path.join(log_dir, f"preprocessing_log_{current_time}.txt")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

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


class GroupSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, groups_per_batch=10, shuffle=True, drop_last=False):
        self.gpb = groups_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last

        buckets = defaultdict(list)
        for idx in range(len(dataset)):
            data = dataset[idx]                     # now walks the entire dataset
            mol_id = int(data.mol_id.item())        # get your group ID
            buckets[mol_id].append(idx)
        self.groups = list(buckets.values())
        sizes = [len(g) for g in self.groups]

        self.n_groups = len(self.groups)
        self.n_short = sum(1 for s in sizes if s < 50)
        self.short_rate = self.n_short / self.n_groups
        self.avg_conf_per_mol = mean(sizes)

    def __iter__(self):
        order = list(range(self.n_groups))
        if self.shuffle:
            random.shuffle(order)
        bucket = []
        for gid in order:
            bucket.extend(self.groups[gid])
            if len(bucket) >= self.gpb * 50:
                yield bucket
                bucket = []
        if bucket and not self.drop_last:
            yield bucket

    def __len__(self):
        full = self.n_groups // self.gpb
        return full if (self.n_groups % self.gpb == 0 or self.drop_last) else full + 1


class E3FPRegressor(nn.Module):
    def __init__(self, bits, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(bits, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.lin = nn.Linear(hidden, 1)

    def forward_conf(self, data):
        h = self.mlp(data.e3fp)
        return h


def make_loaders(processed_dir, batch_size, num_workers, bits):
    train_ds = ConformerDataset(os.path.join(processed_dir, "e3fp_train.pt"))
    val_ds = ConformerDataset(os.path.join(processed_dir, "e3fp_val.pt"))
    train_ds.set_bits(bits)
    val_ds.set_bits(bits)

    gsam_train = GroupSampler(train_ds, groups_per_batch=batch_size, shuffle=True)
    gsam_val = GroupSampler(val_ds, groups_per_batch=batch_size, shuffle=False)

    train_dl = DataLoader(train_ds, batch_sampler=gsam_train,
                          num_workers=num_workers, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_sampler=gsam_val,
                        num_workers=num_workers, pin_memory=False)
    return train_dl, val_dl


def train_epoch(model, loader, optimizer, device, agg):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        conf_reps = model.forward_conf(batch)
        conf_pred = model.lin(conf_reps).view(-1)

        mol_ids = batch.mol_id
        _, inv = torch.unique(mol_ids, sorted=True, return_inverse=True)
        mol_pred = pool_conformers(conf_pred, inv, agg)
        y_true = pool_conformers(batch.y, inv, "mean")

        loss = F.mse_loss(mol_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_true.size(0)
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, device, agg):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            conf_reps = model.forward_conf(batch)
            conf_pred = model.lin(conf_reps).view(-1)

            mol_ids = batch.mol_id
            _, inv = torch.unique(mol_ids, sorted=True, return_inverse=True)
            mol_pred = pool_conformers(conf_pred, inv, agg)
            y_true = pool_conformers(batch.y, inv, "mean")

            loss = F.mse_loss(mol_pred, y_true)
            total_loss += loss.item() * y_true.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--log_dir", required=True)         
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--agg", choices=["sum", "mean", "softmax", "softmin"], default="mean")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = make_loaders(args.processed_dir, args.batch_size, args.num_workers, args.bits)


    model = E3FPRegressor(args.bits, args.hidden).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dl, optimizer, device, args.agg)
        val_loss = validate_epoch(model, val_dl, device, args.agg)
        logging.info(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            # Save the validation score
            with open(args.checkpoint_dir + '/best_val_score.txt', 'w') as f:
                f.write(str(best_val))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info("Early stopping triggered")
                break

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
