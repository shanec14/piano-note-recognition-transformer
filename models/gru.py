import torch
import torch.nn as nn

class GRUNoteClassifier(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim * 2, 88)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.output_layer(self.dropout(gru_out))
        return out
