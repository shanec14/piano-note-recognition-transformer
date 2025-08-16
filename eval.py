import argparse
import os
import json
import yaml
import numpy as np
import torch

from utils.dataset import load_split_data, slice_into_windows, WindowedNoteDataset
from models.gru import GRUNoteClassifier
from models.transformer import TinyTransformer
from models.cnn import CNNMelNoteClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt

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

def get_test_ids(split_csv_path):
    df = pd.read_csv(split_csv_path)
    test_ids = df[df["split"] == "test"]["id"].astype(str).tolist()
    return test_ids

def evaluate_model_at_threshold(model, loader, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(model.device)
            Y_batch = Y_batch.to(model.device)

            if isinstance(model, CNNMelNoteClassifier) and X_batch.ndim == 3:
                X_batch = X_batch.unsqueeze(1)
                Y_batch = (Y_batch.sum(dim=1) > 0).float()

            logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_labels.append(Y_batch.cpu())

    all_logits = torch.cat(all_logits).reshape(-1, 88)
    all_labels = torch.cat(all_labels).reshape(-1, 88).numpy()

    preds = (torch.sigmoid(all_logits) > threshold).float().numpy()

    active_classes = (all_labels.sum(axis=0) + preds.sum(axis=0)) > 0

    active_f1 = f1_score(all_labels[:, active_classes], preds[:, active_classes], average='macro', zero_division=0)
    active_precision = precision_score(all_labels[:, active_classes], preds[:, active_classes], average='macro', zero_division=0)
    active_recall = recall_score(all_labels[:, active_classes], preds[:, active_classes], average='macro', zero_division=0)

    return {
        "threshold": threshold,
        "active_f1": active_f1,
        "active_precision": active_precision,
        "active_recall": active_recall
    }
    
def midi_to_note_name(midi_num):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    note = note_names[midi_num % 12]
    return f"{note}{octave}"


def compute_and_plot_notewise_f1(model, loader, threshold, save_path=None):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(model.device)
            Y_batch = Y_batch.to(model.device)

            if isinstance(model, CNNMelNoteClassifier) and X_batch.ndim == 3:
                X_batch = X_batch.unsqueeze(1)
                Y_batch = (Y_batch.sum(dim=1) > 0).float()

            logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_labels.append(Y_batch.cpu())

    all_logits = torch.cat(all_logits).reshape(-1, 88)
    all_labels = torch.cat(all_labels).reshape(-1, 88).numpy()
    preds = (torch.sigmoid(all_logits) > threshold).float().numpy()

    # Only include active notes (present in labels or predictions)
    f1s_all = f1_score(all_labels, preds, average=None, zero_division=0)
    active_mask = (all_labels.sum(axis=0) + preds.sum(axis=0)) > 0
    f1s = f1s_all[active_mask]

    midi_range = np.arange(21, 109)[active_mask]
    note_labels = [midi_to_note_name(m) for m in midi_range]

    # Plot
    plt.figure(figsize=(20, 6))
    plt.bar(note_labels, f1s)
    plt.xlabel("Note", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.title("Note-wise F1 Score (Active Notes Only) – Transformer (CQT Input)", fontsize=16)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved note-wise F1 score plot to {save_path}")

    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", type=str, help="Model name in hparams.yaml (e.g., GRU, Transformer, CNN)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (e.g., checkpoints/Final/Transformer_220425_134534/best.pth)")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold to apply for evaluation")
    parser.add_argument("--plot_notewise_f1", action="store_true", help="If set, generate and save note-wise F1 score plot")
    args = parser.parse_args()

    # Load config
    with open("hparams.yaml") as f:
        hparams = yaml.safe_load(f)[args.model_key]

    # Load model
    model_name = hparams.get("model_name") 
    model = get_model(model_name, hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.device = device
    model.eval()

    # Load test data
    test_ids = get_test_ids("data_processing/bach_piece_splits.csv")
    X_test, Y_test = load_split_data(hparams["input_type"], test_ids, split="test")
    X_test_windows, Y_test_windows = slice_into_windows(X_test, Y_test, window_size=512, stride=256)
    test_dataset = WindowedNoteDataset(X_test_windows, Y_test_windows)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Evaluate at fixed threshold
    print(f"\nEvaluating model {args.model_key} (loaded from {args.checkpoint}) on test set at threshold {args.threshold:.2f}")
    result = evaluate_model_at_threshold(model, test_loader, threshold=args.threshold)

    # Print results
    print("\n Test Evaluation:")
    print(f"Threshold {result['threshold']:.2f} | Active F1: {result['active_f1']:.4f} | Precision: {result['active_precision']:.4f} | Recall: {result['active_recall']:.4f}")

    # Save result
    model_filename = os.path.basename(args.checkpoint).replace(".pth", "")
    log_filename = f"logs/Eval/{args.model_key}_{model_filename}_test_eval.json"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved test evaluation result to {log_filename}")

    # Optional: plot note-wise F1
    if args.plot_notewise_f1:
        plot_path = f"logs/Eval/{args.model_key}_{model_filename}_f1_per_note.png"
        compute_and_plot_notewise_f1(model, test_loader, threshold=args.threshold, save_path=plot_path)
