import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm
from utils import SimpleLinearRegression
import time 

def train(processed_dir, checkpoint_dir, log_dir, num_epochs=500, learning_rate=1e-3):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write("Training started\n")

    # Load train and val
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))

    # Prepare data
    X_train = torch.tensor(train_df[["2D_TPSA"]].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["permeability"].values, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(val_df[["2D_TPSA"]].values, dtype=torch.float32)
    y_val = torch.tensor(val_df["permeability"].values, dtype=torch.float32).view(-1, 1)

    model = SimpleLinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    total_iterations = num_epochs * len(X_train)

    # Start time for training
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.update(len(X_train))

            # Validate
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val)
                val_loss = criterion(y_val_pred, y_val)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
                log_message = f"Epoch {epoch+1}: Improved validation loss to {best_val_loss:.4f}. Model saved!"
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_message + "\n")
                print(log_message)

            # Log every 100 epochs
            if (epoch + 1) % 100 == 0:
                log_message = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_message + "\n")
                print(log_message)

    # End time for training
    end_time = time.time()

    # Log total training time
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    total_time_message = f"Training completed in {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"
    with open(log_file_path, 'a') as log_file:
        log_file.write(total_time_message + "\n")
    print(total_time_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.processed_dir, args.checkpoint_dir, args.log_dir, args.num_epochs, args.learning_rate)
