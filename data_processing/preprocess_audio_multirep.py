# Imports
from pathlib import Path
from shutil import move
from datetime import datetime
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Constants for MIDI note range (piano)
NOTE_RANGE = list(range(21, 109))  # MIDI notes for full piano range
NOTE_TO_IDX = {note: i for i, note in enumerate(NOTE_RANGE)}  # Map MIDI notes to index


# Converts a list of notes into a multi-hot encoded vector
def multi_hot(notes):
    vec = np.zeros(len(NOTE_RANGE))
    if isinstance(notes, list):
        for note in notes:
            if note in NOTE_TO_IDX:
                vec[NOTE_TO_IDX[note]] = 1
    return vec


# Aligns note labels to frames based on frame duration
def get_frame_labels(df_labels, total_frames, frame_duration):
    labels = [[] for _ in range(total_frames)]  # Initialize empty label list for each frame
    for _, row in df_labels.iterrows():
        s_idx = int(row['start_sec'] / frame_duration)  # Start frame index
        e_idx = int(row['end_sec'] / frame_duration)    # End frame index
        for i in range(s_idx, min(e_idx + 1, total_frames)):
            labels[i].append(row['note'])  # Append note to all frames it spans
    return np.vstack([multi_hot(sorted(set(l))) for l in labels])  # One multi-hot vector per frame


# Processes audio into CQT representation and frame-aligned labels
def process_cqt(y, sr, df_labels):
    hop = 512  # Hop length for frame shift
    cqt = librosa.cqt(y, sr=sr, hop_length=hop, bins_per_octave=12, n_bins=88, fmin=librosa.midi_to_hz(21))
    cqt = librosa.amplitude_to_db(np.abs(cqt)).T  # Convert to dB scale and transpose
    frame_duration = hop / sr  # Duration of one frame
    df_labels['start_sec'] = df_labels['start_time'] / sr  # Convert label times to seconds
    df_labels['end_sec'] = df_labels['end_time'] / sr
    Y = get_frame_labels(df_labels, cqt.shape[0], frame_duration)
    return cqt, Y


# Processes audio into Mel Spectrogram representation and frame-aligned labels
def process_mel(y, sr, df_labels):
    hop = 512
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=128)
    mel = librosa.power_to_db(mel).T  # Convert to dB scale and transpose
    frame_duration = hop / sr
    Y = get_frame_labels(df_labels, mel.shape[0], frame_duration)
    return mel, Y


# Processes audio into Wav2Vec2 features and frame-aligned labels
def process_wav2vec(y, df_labels):
    # Resample audio to 16kHz for Wav2Vec2 model
    y_16k = librosa.resample(y, orig_sr=44100, target_sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    with torch.no_grad():
        inputs = processor(y_16k, sampling_rate=16000, return_tensors="pt", padding=True)
        features = model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
    frame_duration = 320 / 16000  # Frame duration for Wav2Vec2 (20ms)
    Y = get_frame_labels(df_labels, features.shape[0], frame_duration)
    return features, Y


# Saves CQT, Mel, and Wav2Vec2 inputs and their corresponding labels to disk
def save_inputs(base_path, audio_path, csv_path):
    stem = Path(csv_path).stem  # Base filename without extension
    y, sr = librosa.load(audio_path, sr=44100)
    df_labels = pd.read_csv(csv_path)
    df_labels['note'] = df_labels['note'].astype(int)  # Ensure note labels are integers

    X_cqt, Y_cqt = process_cqt(y, sr, df_labels)
    np.save(base_path / f"X_cqt_{stem}.npy", X_cqt)
    np.save(base_path / f"Y_cqt_{stem}.npy", Y_cqt)

    X_mel, Y_mel = process_mel(y, sr, df_labels)
    np.save(base_path / f"X_mel_{stem}.npy", X_mel)
    np.save(base_path / f"Y_mel_{stem}.npy", Y_mel)

    X_wav2vec, Y_wav2vec = process_wav2vec(y, df_labels)
    np.save(base_path / f"X_wav2vec_{stem}.npy", X_wav2vec)
    np.save(base_path / f"Y_wav2vec_{stem}.npy", Y_wav2vec)

    print(f"Saved X/Y for CQT, Mel, and Wav2Vec2 (file: {stem})")


# ====================
# Batch Processing Script
# ====================

# Processes all pieces according to provided train/val/test split
def process_all():
    # Paths
    split_csv = Path("data_processing/bach_piece_splits.csv")
    base_dir = Path("musicnet_dataset/musicnet/musicnet")
    audio_dir = base_dir / "train_data"
    label_dir = base_dir / "train_labels"
    output_base = Path("inputs")
    log_path = Path("logs/Data/preprocessing_log.txt")
    done_path = Path("logs/Data/processed_ids.txt")

    # Create folders for train/val/test if they don't exist
    for split in ['train', 'val', 'test']:
        (output_base / split).mkdir(parents=True, exist_ok=True)

    # Prepare log files
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.touch(exist_ok=True)
    log_path.write_text("")  # Clear old logs

    # Load split assignments and already processed files
    df = pd.read_csv(split_csv).dropna(subset=['id', 'split'])
    done_ids = set(done_path.read_text().splitlines())

    # Iterate over each piece
    for _, row in df.iterrows():
        pid = str(row['id'])  # Piece ID
        split = row['split']  # train/val/test

        if pid in done_ids:
            print(f"Skipping already processed ID {pid}")
            continue

        try:
            # Save inputs for the current piece
            save_inputs(output_base, audio_dir / f"{pid}.wav", label_dir / f"{pid}.csv")

            # Move processed features to corresponding split folder
            for prefix in ['X_cqt_', 'Y_cqt_', 'X_mel_', 'Y_mel_', 'X_wav2vec_', 'Y_wav2vec_']:
                move(str(output_base / f"{prefix}{pid}.npy"), str(output_base / split / f"{prefix}{pid}.npy"))

            # Log success
            msg = f"Finished {pid} - {split} at {datetime.now().strftime('%H:%M:%S')}"
            print(msg)
            log_path.write_text(log_path.read_text() + msg + "\n")

            # Update done list
            done_path.write_text(done_path.read_text() + f"{pid}\n")
            done_ids.add(pid)

        except Exception as e:
            # Handle and log errors
            err_msg = f"*** Error on {pid} at {datetime.now().strftime('%H:%M:%S')}: {e}"
            print(err_msg)
            log_path.write_text(log_path.read_text() + err_msg + "\n")


# Main function to run batch processing
if __name__ == "__main__":
    process_all()
