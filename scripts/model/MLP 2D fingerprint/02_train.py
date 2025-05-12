import os
import torch
import logging
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
from itertools import product
from torch.utils.data import Dataset, DataLoader

class MorganDataset(Dataset):
    def __init__(self, data_list):
        self.X = torch.stack([d['morganfp'] for d in data_list])
        self.y = torch.tensor([d['permeability'] for d in data_list], dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        return self.net(x)

def run_training(processed_dir,
                 num_epochs, learning_rate, batch_size, num_workers,
                 hidden_dim1, hidden_dim2):
    train_data = torch.load(os.path.join(processed_dir, "morganfp_train.pt"))
    val_data   = torch.load(os.path.join(processed_dir, "morganfp_val.pt"))

    train_loader = DataLoader(MorganDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(MorganDataset(val_data), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=train_loader.dataset.X.shape[1],
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = float('inf')
    best_model_state = None  # Store the best model's state_dict
    total_iters = num_epochs * len(train_loader)
    start = time.time()

    with tqdm(total=total_iters, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                pbar.update(X_b.size(0))

            model.eval()
            total_loss = 0
            count = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    l = criterion(model(X_b), y_b).item()
                    total_loss += l * X_b.size(0)
                    count += X_b.size(0)
            val_loss = total_loss / count

            # Track the best model across all epochs
            if val_loss < best_val:
                best_val = val_loss
                best_model_state = model.state_dict()  # Update the best model

            if (epoch+1) % 100 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    elapsed = time.time() - start
    hrs, rem = divmod(elapsed, 3600); mins, secs = divmod(rem, 60)
    logging.info(f"Training completed in {int(hrs)}h {int(mins)}m {int(secs)}s")

    # Save the best model's state and the architecture parameters (hidden_dim1, hidden_dim2)
    model_params = {
        "model_state_dict": best_model_state,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2
    }
    return model_params, best_val  # Return the best model state and validation loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",  type=str, required=True)
    parser.add_argument("--log_dir",     type=str, required=True)
    parser.add_argument("--checkpoint_dir",     type=str, required=True)
    parser.add_argument("--num_epochs",     type=int, default=200)
    parser.add_argument("--num_workers",    type=int, default=0)
    args = parser.parse_args()


    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_path = os.path.join(args.log_dir, "training_log.txt")

    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logging.info("Training started")

    # Grid search settings
    learning_rates = [1e-3, 5e-4]
    hidden_dims1   = [256, 512]
    hidden_dims2   = [64, 128]
    batch_sizes    = [32]

    best_model_state = None
    best_val_loss = float('inf')

    for lr, h1, h2, bs in product(learning_rates, hidden_dims1, hidden_dims2, batch_sizes):
        config_name = f"lr{lr}_h1{h1}_h2{h2}_bs{bs}"
        print(f"\n=== Running config: {config_name} ===")
        best_model_state_curr, best_val_loss_curr = run_training(
            processed_dir=args.processed_dir,
            num_epochs=args.num_epochs,
            learning_rate=lr,
            batch_size=bs,
            num_workers=args.num_workers,
            hidden_dim1=h1,
            hidden_dim2=h2
        )

        # Check if the current configuration has a better validation loss
        if best_val_loss_curr < best_val_loss:
            best_val_loss = best_val_loss_curr
            logging.info(f"best model{config_name}")
            print(f"best model{config_name}")
            best_model_state = best_model_state_curr

    # After the grid search completes, save the best model
    if best_model_state is not None:
        final_checkpoint_dir = os.path.join(args.checkpoint_dir)
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(final_checkpoint_dir, "best_model.pt"))
        logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")
