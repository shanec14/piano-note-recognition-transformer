# Frame-wise Piano Note Classification with Deep Learning

This project explores **Automatic Music Transcription (AMT)** by classifying active piano notes (MIDI 21–108) in polyphonic recordings. The task is framed as a **frame-wise multi-label classification** problem using the [MusicNet dataset](https://zenodo.org/record/5120004).

Three models were implemented from scratch in PyTorch:

- **CNN** on Mel-spectrograms (baseline)  
- **Bidirectional GRU** on CQT and Wav2Vec2 embeddings  
- **Transformer Encoder** on CQT and Wav2Vec2 embeddings  

Full details are provided in the [Project Report](Project_report.pdf).

---

## Project Overview
- **Goal**: Convert raw piano audio into symbolic note activations.  
- **Dataset**: 39 solo Bach pieces from MusicNet.  
- **Inputs**:  
  - Constant-Q Transform (CQT, 88 bins)  
  - Wav2Vec2 embeddings (768 dims, 50 fps)  
  - Mel-spectrogram (for CNN baseline)  
- **Outputs**: Frame-wise binary activations for 88 piano keys.  

---

## Models

- **CNNMelNoteClassifier** ([`cnn.py`](cnn.py))  
  2D CNN with global average pooling and fully connected layers.  

- **GRUNoteClassifier** ([`gru.py`](gru.py))  
  2–3 layer bidirectional GRU with dropout and linear projection.  

- **TinyTransformer** ([`transformer.py`](transformer.py))  
  Lightweight Transformer Encoder with positional encoding, multi-head self-attention, and feedforward layers.  


---

## Results

| Model-Input             | F1 Score | Precision | Recall |
|-------------------------|----------|-----------|--------|
| CNN (Mel)              | 0.63     | 0.61      | 0.70   |
| GRU (CQT)              | 0.63     | 0.66      | 0.62   |
| GRU (Wav2Vec2)         | 0.29     | 0.30      | 0.31   |
| Transformer (CQT)      | **0.68** | 0.72      | 0.67   |
| Transformer (Wav2Vec2) | 0.31     | 0.34      | 0.34   |

**Key insight**:  
- The **Transformer with CQT** achieved the best results (Active F1 = 0.68), outperforming the CNN baseline by 8%.  
- Wav2Vec2 embeddings underperformed due to speech-centric pretraining and lower temporal resolution.  

---

## Setup

### Requirements
- Python 3.9+
- PyTorch >= 1.13
- torchaudio
- librosa
- numpy, matplotlib, tqdm



## Dataset

Download [MusicNet](https://zenodo.org/record/5120004) and place `.wav` audio files and `.csv` annotation files under `data/`.

Preprocessing extracts:
- CQT representations  
- Wav2Vec2 embeddings  
- Frame-wise multi-hot labels  

---

## Usage

Train a model:
```
python train.py --model cnn          # CNN baseline
python train.py --model gru          # GRU on CQT or Wav2Vec2
python train.py --model transformer  # Transformer on CQT or Wav2Vec2

```
Got it — here’s the whole thing in proper Markdown format, ready to paste directly into any README.md or Markdown environment:

## Evaluate

```
python evaluate.py --model transformer --input cqt
```

(Training scripts follow the pipeline described in the report; adapt paths and parameters as needed.)


Future Work

- Fine-tune Wav2Vec2 on music data for better pitch alignment.
- Explore hybrid CQT + learned embeddings.
- Apply onset-aware models (e.g., Onsets & Frames).
- Data augmentation for rare and high-pitch notes.

References

- Hawthorne et al. Onsets and Frames: Dual-Objective Piano Transcription. ISMIR, 2018.
- Jamshidi et al. Recent Advances in Automatic Music Transcription. IEEE SPM, 2024.
- Ou et al. Exploring Transformer’s Potential on Automatic Piano Transcription. arXiv, 2022.
- Baevski et al. wav2vec 2.0: A framework for self-supervised learning of speech representations. NeurIPS, 2020.
- Brown, J. C. Calculation of a Constant Q Spectral Transform. JASA, 1991.


