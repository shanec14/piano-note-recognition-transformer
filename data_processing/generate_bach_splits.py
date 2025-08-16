import pandas as pd
import os
import random
from pathlib import Path

# CONFIGURATION
BASE_PATH = Path("musicnet_dataset")  # Base directory for dataset
ANNOTATION_DIR = BASE_PATH / "musicnet/musicnet" / "train_labels"  # Directory containing note annotations
METADATA_PATH = BASE_PATH / "musicnet_metadata.csv"  # Path to metadata CSV
OUTPUT_CSV = "data_processing/bach_piece_splits.csv"  # Path to save/load train/val/test splits

# Split sizes
N_TRAIN = 25
N_VAL = 7
N_TEST = 7
MAX_ATTEMPTS = 100  # Max attempts to find a valid split

# Load metadata about pieces
metadata = pd.read_csv(METADATA_PATH)

# Filter metadata to only include solo piano pieces composed by Bach
bach_solo_df = metadata[
    metadata['composer'].str.contains("Bach", na=False) &  # Composer is Bach
    metadata['ensemble'].str.lower().str.contains("solo", na=False) &  # Ensemble contains "solo"
    metadata['ensemble'].str.lower().str.contains("piano", na=False)  # Ensemble contains "piano"
]

# List of IDs for Bach solo piano pieces
bach_ids = bach_solo_df['id'].tolist()

# Function to load notes present in each annotated piece
def load_notes(annotation_dir):
    notes = {}
    for file in annotation_dir.glob("*.csv"):  # Loop through all annotation files
        piece_id = int(file.stem)  # Extract piece ID from filename
        df = pd.read_csv(file)  # Read note annotations
        notes[piece_id] = set(df['note'].values)  # Store unique notes for the piece
    return notes

# Load note data from annotation files
notes_from_piece = load_notes(ANNOTATION_DIR)

# Keep only Bach IDs that actually have annotations
bach_ids = [pid for pid in bach_ids if pid in notes_from_piece]

# Either load an existing split or create a new one
if Path(OUTPUT_CSV).exists():
    # If split already exists, load it
    print(f"Split file '{OUTPUT_CSV}' already exists. Loading existing split.")
    split_df = pd.read_csv(OUTPUT_CSV)
    train_ids = split_df[split_df['split'] == 'train']['id'].tolist()
    val_ids = split_df[split_df['split'] == 'val']['id'].tolist()
    test_ids = split_df[split_df['split'] == 'test']['id'].tolist()
else:
    # Otherwise, create a new train/val/test split
    print(f"Split file '{OUTPUT_CSV}' not found. Generating new split...")

    # Helper function to get the union of notes across multiple pieces
    def get_all_notes(ids):
        return set.union(*[notes_from_piece[i] for i in ids])

    # Try to create a split where all validation and test notes are covered by training set
    for attempt in range(1, MAX_ATTEMPTS + 1):
        random.shuffle(bach_ids)  # Randomize the order
        train_ids = bach_ids[:N_TRAIN]
        val_ids = bach_ids[N_TRAIN:N_TRAIN + N_VAL]
        test_ids = bach_ids[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

        train_notes = get_all_notes(train_ids)
        val_notes = get_all_notes(val_ids)
        test_notes = get_all_notes(test_ids)

        # Check coverage: val/test notes must be subsets of train notes
        if val_notes.issubset(train_notes) and test_notes.issubset(train_notes):
            print(f"Valid split found on attempt {attempt}")
            break
    else:
        # If no valid split is found after max attempts, raise an error
        raise ValueError("Could not find a valid split after 100 tries")

    # Save the successful split to CSV
    split_df = pd.DataFrame({
        'id': train_ids + val_ids + test_ids,
        'split': ['train'] * len(train_ids) + ['val'] * len(val_ids) + ['test'] * len(test_ids)
    })
    split_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved new split to {OUTPUT_CSV}")
