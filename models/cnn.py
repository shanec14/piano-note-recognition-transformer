import torch
import torch.nn as nn

class CNNMelNoteClassifier(nn.Module):
    def __init__(self, input_channels=1, num_notes=88):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.gap = nn.AdaptiveAvgPool2d((8, 11))
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 11, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_notes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.gap(x)
        x = self.fc_stack(x)
        return x