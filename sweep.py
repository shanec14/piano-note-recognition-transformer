import yaml
import os
import subprocess

# Full experimental sweep configuration based on the provided CSV structure
configs = [
    # GRU on CQT
    {"model_name": "gru", "input_type": "cqt", "model_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 128, "num_layers": 3, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 128, "num_layers": 3, "dropout": 0.3, "lr": 5e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 256, "num_layers": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 256, "num_layers": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 256, "num_layers": 3, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "gru", "input_type": "cqt", "model_dim": 256, "num_layers": 3, "dropout": 0.3, "lr": 5e-4},

    # # Transformer on CQT
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 128, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 128, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 128, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 128, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 5e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 256, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 256, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 256, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "cqt", "model_dim": 256, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 5e-4},

    # GRU on Wav2Vec2
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 128, "num_layers": 3, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 128, "num_layers": 3, "dropout": 0.3, "lr": 5e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 256, "num_layers": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 256, "num_layers": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 256, "num_layers": 3, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "gru", "input_type": "wav2vec", "model_dim": 256, "num_layers": 3, "dropout": 0.3, "lr": 5e-4},

    # Transformer on Wav2Vec2
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 128, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 128, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 128, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 128, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 5e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 256, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 256, "num_layers": 2, "num_heads": 2, "dropout": 0.1, "lr": 5e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 256, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 1e-4},
    {"model_name": "transformer", "input_type": "wav2vec", "model_dim": 256, "num_layers": 3, "num_heads": 4, "dropout": 0.3, "lr": 5e-4},
]

template = {"default": {"num_epochs": 500, "batch_size": 128, "patience": 15, "early_stopping": True, "use_fraction": 1.0, "min_f1_threshold": 0.2,"min_f1_epochs": 500}}

# Assign keys and merge defaults
# for i, cfg in enumerate(configs):
for i, cfg in enumerate(configs, start=0):
    key = f"{cfg['model_name']}_{i}"
    template[key] = {**template["default"], **cfg, "gpu_vis_dev": "0"}

# Save to temp YAML
with open("experiment_hparams.yaml", "w") as f:
    yaml.dump(template, f)
    
    


for key in template:
    if key == "default":
        continue
    cmd = f"python main.py {key} --hparams experiment_hparams.yaml"
    print(f"Launching: {cmd}")
    subprocess.run(cmd, shell=True)

