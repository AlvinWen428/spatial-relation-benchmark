from typing import List
import torch
import torch.nn as nn


class PromptReadoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.0):
        super(PromptReadoutNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
