"""Downstream classifier on top of TFNFullModel.

Takes the last-timestep hidden state and adds an MLP head for binary classification.
The backbone is frozen.
"""
import torch.nn as nn


class TFNDownstreamClassifier(nn.Module):
    def __init__(self, backbone_model, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone_model
        n_embd = backbone_model.n_embd
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_embd // 2, num_classes),
        )

    def forward(self, input_ids, elapsed_times=None, static_categorical=None, static_numerical=None):
        out = self.backbone(
            input_ids=input_ids,
            elapsed_times=elapsed_times,
            static_categorical=static_categorical,
            static_numerical=static_numerical,
            labels=None,
        )
        # Last-timestep representation: (B, n_embd)
        pooled = out.hidden_states[:, -1, :]
        logits = self.classifier(pooled)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits
