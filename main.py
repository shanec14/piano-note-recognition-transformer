import os
import time
import json
import argparse
import importlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from utils.dataset import load_split_data, slice_into_windows, WindowedNoteDataset
from utils.metrics import evaluate_model_threshold_sweep

import yaml
import pandas as pd

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_params(hparams_path, model_key):
    """Load hyperparameters from a YAML file given a model key."""
    with open(hparams_path) as f:
        all_params = yaml.safe_load(f)
    return all_params[model_key]

def get_model_instance(model_key, params):
    """Dynamically import and instantiate the selected model based on hyperparameters."""
    model_name = params["model_name"].lower()
    model_module = importlib.import_module(f"models.{model_name}")

    input_type = params["input_type"]
    input_dims = {"cqt": 88, "wav2vec": 768}
    input_dim = input_dims.get(input_type, 88)

    if model_name == "cnn":
        return model_module.CNNMelNoteClassifier()
    elif model_name == "gru":
        return model_module.GRUNoteClassifier(input_dim=input_dim,
                                              hidden_dim=params["model_dim"],
                                              num_layers=params["num_layers"],
                                              dropout=params["dropout"])
    elif model_name == "transformer":
        return model_module.TinyTransformer(input_dim=input_dim,
                                            model_dim=params["model_dim"],
                                            num_heads=params["num_heads"],
                                            num_layers=params["num_layers"],
                                            output_dim=88,
                                            dropout=params["dropout"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_ids_from_split(split_csv_path):
    """Load train and validation IDs from a CSV split file."""
    df = pd.read_csv(split_csv_path)
    train_ids = df[df["split"] == "train"]["id"].astype(str).tolist()
    val_ids = df[df["split"] == "val"]["id"].astype(str).tolist()
    return train_ids, val_ids

def compute_pos_weight(Y_tensor):
    """Compute positive class weights for BCEWithLogitsLoss."""
    note_freq = Y_tensor.sum(dim=0)
    total_frames = Y_tensor.shape[0]
    pos_weight = (total_frames - note_freq) / (note_freq + 1e-5)
    pos_weight = torch.where(note_freq > 0, pos_weight, torch.zeros_like(pos_weight))
    return pos_weight

def main():
    # Set seed for full reproducibility
    set_seed(2025)

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", type=str, help="Model key name (e.g., gru_0)")
    parser.add_argument("--hparams", type=str, default="hparams.yaml", help="Path to hyperparameter YAML file")
    args = parser.parse_args()

    # Load hyperparameters
    params = load_params(args.hparams, args.model_key)

    # Set GPU device visibility
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_vis_dev"]

    # Create unique run ID based on timestamp
    start_time = time.strftime("%d%m%y_%H%M%S")
    run_id = f"{args.model_key}_{start_time}"

    # Setup directories
    checkpoint_dir = os.path.join("checkpoints", "Final", args.model_key + "_" + start_time)
    log_dir = os.path.join("logs", "Final", args.model_key)
    log_path = os.path.join(log_dir, f"{run_id}.json")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_instance(args.model_key, params).to(device)
    model.device = device

    # Load data splits
    train_ids, val_ids = get_ids_from_split("data_processing/bach_piece_splits.csv")
    X_train, Y_train = load_split_data(params["input_type"], train_ids, split="train", use_fraction=params.get("use_fraction", 1.0))
    X_val, Y_val = load_split_data(params["input_type"], val_ids, split="val")

    # Slice sequences
    X_train, Y_train = slice_into_windows(X_train, Y_train, window_size=512, stride=256)
    X_val, Y_val = slice_into_windows(X_val, Y_val, window_size=512, stride=256)

    # Create DataLoaders
    train_loader = DataLoader(WindowedNoteDataset(X_train, Y_train), batch_size=params["batch_size"],
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(WindowedNoteDataset(X_val, Y_val), batch_size=params["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    # Initialise optimiser
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # Initialise loss function
    Y_train_tensor = torch.tensor(Y_train).reshape(-1, 88)
    if params.get("pos_weight") == "auto":
        print("Using automatic pos_weight based on training labels")
        pos_weight = compute_pos_weight(Y_train_tensor)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    elif params.get("pos_weight") == "clamp100":
        print("Using clamped pos_weight (max 100)")
        pos_weight = compute_pos_weight(Y_train_tensor)
        pos_weight = torch.clamp(pos_weight, max=100)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        print("Using normal BCEWithLogitsLoss")
        loss_fn = torch.nn.BCEWithLogitsLoss()

    scaler = GradScaler()

    # Tracking
    train_losses, val_f1s = [], []
    best_f1 = -float("inf")
    patience, epochs_no_improve = params.get("patience", 10), 0
    early_stopping = params["early_stopping"]
    min_f1_threshold = params.get("min_f1_threshold", 0.1)
    min_f1_epochs = params.get("min_f1_epochs", 50)

    # Training loop
    for epoch in range(params["num_epochs"]):
        model.train()
        running_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if args.model_key.lower().startswith("cnn"):
                X_batch = X_batch.unsqueeze(1)
                Y_batch = (Y_batch.sum(dim=1) > 0).float()

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits = model(X_batch)
                loss = loss_fn(logits, Y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        val_metrics = evaluate_model_threshold_sweep(model, val_loader, thresholds=[0.3])[0]
        val_f1 = val_metrics["active_f1"]
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Active F1={val_f1:.4f}")

        if epoch < min_f1_epochs and val_f1 < min_f1_threshold:
            continue

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    # Save training logs
    hparam_log = {k: v for k, v in params.items() if k != "gpu_vis_dev"}
    hparam_log.update({
        "model": args.model_key,
        "run_id": run_id,
        "train_losses": train_losses,
        "val_f1s": val_f1s,
        "best_val_epoch": int(np.argmax(val_f1s) + 1),
        "best_val_f1": float(np.max(val_f1s))
    })

    with open(log_path, "w") as f:
        json.dump(hparam_log, f, indent=2)

if __name__ == '__main__':
    main()
