import os
import torch,logging
import pandas as pd
import argparse
from datetime import datetime
import time
from sklearn.metrics import r2_score, mean_squared_error

# Re-define the MLP to match the training model
import torch.nn as nn
class MLP(nn.Module):
    """
    A simple MLP that maps Morgan fingerprint vectors to a scalar permeability.
    """
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=264):
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

def predict(processed_dir, checkpoint_dir, result_dir, log_dir):
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"prediction_log_{current_time}.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write("Prediction started\n")

    start_time = time.time()

    # Load the trained model and its parameters
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path)
    
    hidden_dim1 = checkpoint["hidden_dim1"]
    hidden_dim2 = checkpoint["hidden_dim2"]

    # Load test data and infer input_dim
    test_data = torch.load(os.path.join(processed_dir, "morganfp_test.pt"))
    if len(test_data) == 0:
        raise ValueError("No test examples found in processed_dir")
    input_dim = test_data[0]['morganfp'].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare inputs and ground-truth
    sequences = [d['sequence'] for d in test_data]
    y_true = [d['permeability'] for d in test_data]
    X = torch.stack([d['morganfp'] for d in test_data]).to(device)

    # Run prediction
    with torch.no_grad():
        y_pred = model(X).cpu().numpy().flatten()

    # Save results
    result_df = pd.DataFrame({
        'sequence': sequences,
        'predict_permeability': y_pred,
        'true_permeability': y_true
    })
    save_path = os.path.join(result_dir, "test_predict.csv")
    result_df.to_csv(save_path, index=False)

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    end_time = time.time()
    elapsed = end_time - start_time

    # Log messages
    log_messages = [
        f"Saved predictions to {save_path}",
        f"R2 Score: {r2:.4f}",
        f"Mean Squared Error (MSE): {mse:.4f}",
        f"Prediction time: {elapsed:.2f} seconds"
    ]
    with open(log_file_path, 'a') as log_file:
        for msg in log_messages:
            log_file.write(msg + "\n")
    for msg in log_messages:
        print(msg)
        logging.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    predict(
        args.processed_dir,
        args.checkpoint_dir,
        args.result_dir,
        args.log_dir

    )
