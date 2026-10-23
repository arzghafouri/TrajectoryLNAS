"""
Abstract Base Trainer for Trajectory Prediction Models
======================================================
Provides standard lifecycle methods, device resolution, and checkpoint handling.
"""

from abc import ABC, abstractmethod
import torch
import os

class BaseTrainer(ABC):
    """
    Abstract Base Class for training, searching, and validating models.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        else:
            self.device = device

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        """Runs one training epoch over the training data."""
        pass

    @abstractmethod
    def validate(self, epoch: int) -> dict:
        """Runs evaluation over the validation set."""
        pass

    def save_checkpoint(self, state_dict: dict, file_path: str):
        """Saves model weights safely, creating parent folders if needed."""
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        torch.save(state_dict, file_path)
        print(f"[Checkpoint] Successfully saved to: {file_path}")

    def load_checkpoint(self, file_path: str) -> dict:
        """Loads checkpoint from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        return torch.load(file_path, map_location=self.device)
