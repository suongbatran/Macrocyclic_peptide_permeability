import os
import torch
import pandas as pd
import argparse
from utils import SimpleLinearRegression
from datetime import datetime
import time
from sklearn.metrics import r2_score, mean_squared_error

def predict(processed_dir, checkpoint_dir, result_dir, log_dir):
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"prediction_log_{current_time}.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write("Prediction started\n")

    start_time = time.time()  # Start timing

    model = SimpleLinearRegression()
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt")))
    model.eval()

    test_file = "test.csv"
    path = os.path.join(processed_dir, test_file)
    df = pd.read_csv(path)

    X = torch.tensor(df[["2D_TPSA"]].values, dtype=torch.float32)
    y_true = df["permeability"].values  # Ground truth labels

    predictions = model(X).detach().numpy().flatten()

    result_df = df[["sequence"]].copy()
    result_df["predict_permeability"] = predictions

    save_path = os.path.join(result_dir, f"{test_file.replace('.csv', '')}_predict.csv")
    result_df.to_csv(save_path, index=False)

    # Calculate metrics
    r2 = r2_score(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time

    # Log messages
    log_messages = [
        f"Saved prediction to {save_path}",
        f"R2 Score: {r2:.4f}",
        f"Mean Squared Error (MSE): {mse:.4f}",
        f"Time taken for prediction: {elapsed_time:.2f} seconds"
    ]

    with open(log_file_path, 'a') as log_file:
        for message in log_messages:
            log_file.write(message + "\n")

    for message in log_messages:
        print(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()

    predict(args.processed_dir, args.checkpoint_dir, args.result_dir, args.log_dir)
