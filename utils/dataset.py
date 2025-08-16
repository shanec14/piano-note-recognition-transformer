import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Loads and prepares data for a specific train/val/test split.
def load_split_data(input_type, ids, split="train", base_path="inputs", use_fraction=1.0, seed=42):
    X_list, Y_list = [], []

    # Optionally reduce dataset size (for faster debugging/training)
    if use_fraction < 1.0:
        np.random.seed(seed)
        n_keep = int(use_fraction * len(ids))
        ids = sorted(ids)[:n_keep]

    split_path = os.path.join(base_path, split)

    for audio_id in ids:
        # Load saved feature (X) and label (Y) arrays for each audio piece
        X = np.load(os.path.join(split_path, f"X_{input_type}_{audio_id}.npy"))
        Y = np.load(os.path.join(split_path, f"Y_{input_type}_{audio_id}.npy"))

        # Normalize input features to zero mean and unit variance (per audio piece)
        X = (X - X.mean()) / (X.std() + 1e-8)

        X_list.append(X)
        Y_list.append(Y)

    # Concatenate all pieces into one big array
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)

    return X_all, Y_all

# Splits input (X) and label (Y) sequences into overlapping sliding windows.
def slice_into_windows(X, Y, window_size=2048, stride=1024, start=0):
    X_chunks, Y_chunks = [], []
    T = X.shape[0]  # Total number of frames

    # Slide a window across the sequence
    for i in range(start, T - window_size + 1, stride):
        X_chunks.append(X[i:i+window_size])
        Y_chunks.append(Y[i:i+window_size])

    # Stack all chunks into arrays
    return np.stack(X_chunks), np.stack(Y_chunks)

# A PyTorch dataset class for windowed note classification.
class WindowedNoteDataset(Dataset):
    def __init__(self, X_chunks, Y_chunks):
        # Store features and labels as float32 tensors
        self.X = torch.tensor(X_chunks, dtype=torch.float32)
        self.Y = torch.tensor(Y_chunks, dtype=torch.float32)

    def __len__(self):
        # Return total number of windows
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return feature and label window at the given index
        return self.X[idx], self.Y[idx]
