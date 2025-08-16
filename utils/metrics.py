from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
from models.cnn import CNNMelNoteClassifier

def evaluate_model_threshold_sweep(model, loader, thresholds=[0.5]):
    """
    Evaluates a multi-label classification model (frame-wise piano note prediction)
    across a range of thresholds.

    Args:
        model: Trained PyTorch model to evaluate.
        loader: DataLoader providing (X_batch, Y_batch) pairs.
        thresholds: List of decision thresholds to test (default 0.5).

    Returns:
        A list of dictionaries containing evaluation metrics for each threshold.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    all_logits, all_labels = [], []  # Store outputs and ground truth labels

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for X_batch, Y_batch in loader:
            # Move batch to correct device (CPU or GPU)
            X_batch = X_batch.to(model.device)
            Y_batch = Y_batch.to(model.device)

            # Special handling for CNNMelNoteClassifier:
            # - Add a channel dimension if missing
            # - Merge multiple note labels into a single label per frame
            if isinstance(model, CNNMelNoteClassifier) and X_batch.ndim == 3:
                X_batch = X_batch.unsqueeze(1)
                Y_batch = (Y_batch.sum(dim=1) > 0).float()

            # Forward pass: compute model outputs (logits)
            logits = model(X_batch)
            all_logits.append(logits.cpu())  # Store outputs on CPU
            all_labels.append(Y_batch.cpu())  # Store ground truths on CPU

    # Concatenate all stored outputs and labels into full tensors
    all_logits = torch.cat(all_logits).reshape(-1, 88)  # Flatten to (frames, notes)
    all_labels = torch.cat(all_labels).reshape(-1, 88).numpy()

    results = []

    # Sweep across all specified thresholds
    for thresh in thresholds:
        # Apply sigmoid activation and thresholding to produce binary predictions
        preds = (torch.sigmoid(all_logits) > thresh).float().numpy()

        # Calculate evaluation metrics
        macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        macro_precision = precision_score(all_labels, preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)

        # "Active classes" are notes that actually occur at least once
        active_classes = (all_labels.sum(axis=0) + preds.sum(axis=0)) > 0

        # Calculate metrics only on active classes (ignoring never-used labels)
        active_f1 = f1_score(all_labels[:, active_classes], preds[:, active_classes], average='macro', zero_division=0)
        active_precision = precision_score(all_labels[:, active_classes], preds[:, active_classes], average='macro', zero_division=0)
        active_recall = recall_score(all_labels[:, active_classes], preds[:, active_classes], average='macro', zero_division=0)

        # Save results for this threshold
        results.append({
            "threshold": thresh,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "weighted_f1": weighted_f1,
            "active_f1": active_f1,
            "active_precision": active_precision,
            "active_recall": active_recall
        })

        # Print quick summary for this threshold
        print(f"[Threshold = {thresh:.2f}] Active F1: {active_f1:.4f} | Precision: {active_precision:.4f} | Recall: {active_recall:.4f}")

    return results



def threshold_sweep(model_key, checkpoint_path, split="val", thresholds=np.arange(0.05, 0.95, 0.05)):
    """
    Automatically load model and data, sweep thresholds, return best one.
    """

    import torch
    import yaml
    import pandas as pd
    import numpy as np
    from utils.dataset import load_split_data, slice_into_windows, WindowedNoteDataset
    from models.gru import GRUNoteClassifier
    from models.transformer import TinyTransformer
    from models.cnn import CNNMelNoteClassifier

    # Load hparams
    with open("hparams.yaml") as f:
        hparams = yaml.safe_load(f)[model_key]

    # Build model
    def get_model(model_key, params):
        if model_key.lower() == "gru":
            return GRUNoteClassifier(
                input_dim=params["input_dim"],
                hidden_dim=params["model_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            )
        elif model_key.lower() == "transformer":
            return TinyTransformer(
                input_dim=params["input_dim"],
                model_dim=params["model_dim"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                output_dim=88,
                dropout=params["dropout"]
            )
        elif model_key.lower() == "cnn":
            return CNNMelNoteClassifier()
        else:
            raise ValueError(f"Unknown model key: {model_key}")

    model = get_model(hparams["model_name"], hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.device = device
    model.eval()

    # Load split data (validation or test)
    split_csv_path = "data_processing/bach_piece_splits.csv"
    df = pd.read_csv(split_csv_path)
    ids = df[df["split"] == split]["id"].astype(str).tolist()

    X, Y = load_split_data(hparams["input_type"], ids, split=split)
    X_windows, Y_windows = slice_into_windows(X, Y, window_size=512, stride=256)
    dataset = WindowedNoteDataset(X_windows, Y_windows)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    # Sweep thresholds
    results = evaluate_model_threshold_sweep(model, loader, thresholds)

    # Get best threshold
    best_result = max(results, key=lambda r: r["active_f1"])
    best_threshold = best_result["threshold"]

    return results, best_threshold
