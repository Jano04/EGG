# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json
from pathlib import Path
from typing import Dict, List

from egg.core import Callback, Interaction


class ResetAlternationCounter(Callback):
    """
    Callback that resets the batch counter at the start of each epoch.
    This ensures every epoch starts with A→B direction for predictable alternation.

    Required for games using AlternatingGame wrapper to maintain consistent
    alternation patterns across epochs.
    """
    def on_epoch_begin(self, epoch: int):
        if hasattr(self.trainer.game, 'reset_batch_counter'):
            self.trainer.game.reset_batch_counter()


class DirectionalConsoleLogger(Callback):
    """
    Custom logger that separates and displays metrics by communication direction.
    Shows both A→B and B→A accuracies separately, plus the average.

    Designed for role-alternating games where agents swap between sender and receiver.
    Expects the game to tag interactions with 'direction' field in aux dict:
        - direction == 0: A→B communication
        - direction == 1: B→A communication

    Args:
        n_epochs: Total number of training epochs (for display formatting)
        print_train_loss: Whether to print training loss after each epoch
    """
    def __init__(self, n_epochs: int, print_train_loss: bool = True):
        self.n_epochs = n_epochs
        self.print_train_loss = print_train_loss

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self._print_directional_stats(loss, logs, epoch, mode="train")

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self._print_directional_stats(loss, logs, epoch, mode="val")

    def _print_directional_stats(self, loss: float, logs: Interaction, epoch: int, mode: str):
        """Compute and print metrics separated by direction."""
        if 'direction' not in logs.aux:
            # Fallback if no direction info
            avg_acc = logs.aux.get('acc', torch.tensor(0)).mean().item()
            print(f"Epoch {epoch} ({mode}): Loss={loss:.4f}, Acc={avg_acc:.3f}")
            return

        direction = logs.aux['direction']

        # Separate A→B (direction==0) and B→A (direction==1)
        mask_AB = (direction == 0)
        mask_BA = (direction == 1)

        # Calculate metrics for each direction
        metrics_AB = {}
        metrics_BA = {}
        for key, values in logs.aux.items():
            if key == 'direction':
                continue
            if mask_AB.any():
                metrics_AB[key] = values[mask_AB].mean().item()
            if mask_BA.any():
                metrics_BA[key] = values[mask_BA].mean().item()

        # Print formatted output
        if mode == "train":
            acc_AB = metrics_AB.get('acc', 0)
            acc_BA = metrics_BA.get('acc', 0)
            avg_acc = (acc_AB + acc_BA) / 2 if (acc_AB and acc_BA) else (acc_AB or acc_BA)
            print(f"Epoch {epoch:3}/{self.n_epochs} - "
                  f"Loss: {loss:.4f}, Acc: {avg_acc:.3f} "
                  f"[A→B: {acc_AB:.3f}, B→A: {acc_BA:.3f}]")
        else:  # validation
            acc_AB = metrics_AB.get('acc', 0)
            acc_BA = metrics_BA.get('acc', 0)
            avg_acc = (acc_AB + acc_BA) / 2 if (acc_AB and acc_BA) else (acc_AB or acc_BA)
            print(f"  ├─ Validation A→B: Loss={loss:.4f}, Acc={acc_AB:.3f}")
            print(f"  ├─ Validation B→A: Loss={loss:.4f}, Acc={acc_BA:.3f}")
            print(f"  └─ Average:        Loss={loss:.4f}, Acc={avg_acc:.3f}\n")
