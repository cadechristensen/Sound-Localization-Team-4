"""
Training module for baby cry detection model.
Implements training loop with validation, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

try:
    from .config import Config
    from .model import create_model, count_parameters
    from .dataset import DatasetManager
except ImportError:
    import sys
    from pathlib import Path as PathLib
    src_dir = PathLib(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import Config  # type: ignore
    from model import create_model, count_parameters  # type: ignore
    from dataset import DatasetManager  # type: ignore


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    Focuses learning on hard-to-classify examples (like weak cries).

    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        alpha: Weighting factor for class balance (0-1)
        gamma: Focusing parameter for hard examples (typically 2.0)
        weight: Class weights tensor
    """

    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last time validation loss improved
            min_delta: Minimum change in validation loss to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False

    def save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    def restore_checkpoint(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict({key: value.to(next(model.parameters()).device)
                                 for key, value in self.best_weights.items()})


class BabyCryTrainer:
    """
    Trainer class for baby cry detection model.
    Handles training, validation, and model checkpointing.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize the trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

        # Initialize model
        self.model = create_model(config).to(self.device)
        logging.info(f"Model initialized with {count_parameters(self.model):,} trainable parameters")

        # Initialize dataset manager
        self.dataset_manager = DatasetManager(config)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

    def setup_training(self):
        """Set up training components."""
        # Prepare datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.prepare_datasets()

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self.dataset_manager.create_data_loaders(
            self.train_dataset, self.val_dataset, self.test_dataset
        )

        # Set up loss function with Focal Loss (better for hard examples like weak cries)
        class_weights = self.dataset_manager.class_weights.to(self.device)
        focal_alpha = getattr(self.config, 'FOCAL_LOSS_ALPHA', 0.25)
        focal_gamma = getattr(self.config, 'FOCAL_LOSS_GAMMA', 2.0)
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=class_weights)
        logging.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}) with cry_weight_multiplier={getattr(self.config, 'CRY_WEIGHT_MULTIPLIER', 2.0)} for enhanced weak cry detection")

        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Set up learning rate scheduler (more patient, smaller reductions)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,  # Smaller reduction (was 0.5)
            patience=7,  # More patience (was 5)
            min_lr=1e-6,
            threshold=0.001  # Only reduce if improvement < 0.1%
        )

        # Set up early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            min_delta=0.001
        )

        logging.info("Training setup completed")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch_data in enumerate(pbar):
            # Handle both old format (specs, labels) and new format (specs, labels, indices)
            if len(batch_data) == 3:
                spectrograms, labels, _ = batch_data
            else:
                spectrograms, labels = batch_data

            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(spectrograms)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability (more conservative)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # Update weights
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # Handle both old format (specs, labels) and new format (specs, labels, indices)
                if len(batch_data) == 3:
                    spectrograms, labels, _ = batch_data
                else:
                    spectrograms, labels = batch_data

                spectrograms = spectrograms.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def train(self, results_dir: Path) -> Dict:
        """
        Main training loop.

        Args:
            results_dir: Directory to save results

        Returns:
            Training history dictionary
        """
        logging.info("Starting training...")
        start_time = time.time()

        best_val_loss = float('inf')
        best_model_path = results_dir / "model_best.pth"

        # Track last N checkpoints for ensembling
        last_n_checkpoints = []
        n_checkpoints_to_keep = 3

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()

            # Training phase
            train_loss, train_acc = self.train_epoch()

            # Validation phase
            val_loss, val_acc = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config.__dict__
                }, best_model_path)

            # Save last N checkpoints for ensembling (in last 10 epochs)
            if epoch >= self.config.NUM_EPOCHS - 10:
                checkpoint_path = results_dir / f"model_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config.__dict__
                }, checkpoint_path)

                last_n_checkpoints.append((checkpoint_path, val_loss))

                # Keep only best N checkpoints
                if len(last_n_checkpoints) > n_checkpoints_to_keep:
                    # Sort by validation loss
                    last_n_checkpoints.sort(key=lambda x: x[1])
                    # Remove worst checkpoint
                    worst_checkpoint = last_n_checkpoints.pop()
                    if worst_checkpoint[0].exists():
                        worst_checkpoint[0].unlink()
                        logging.info(f"Removed checkpoint: {worst_checkpoint[0].name}")

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Logging
            logging.info(
                f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.4f} "
                f"LR: {current_lr:.6f} "
                f"Time: {epoch_time:.2f}s"
            )

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

        # Training completed
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")

        # Save training history
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                    f"with validation loss {checkpoint['val_loss']:.4f}")

        return checkpoint

    def save_model_for_inference(self, results_dir: Path):
        """
        Save optimized model for inference.

        Args:
            results_dir: Directory to save the model
        """
        # Set model to evaluation mode
        self.model.eval()

        # Save inference model (optimized, no training artifacts)
        inference_path = results_dir / "model_inference.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'class_labels': self.config.CLASS_LABELS
        }, inference_path)

        logging.info(f"Inference model saved to {inference_path}")


def setup_logging(results_dir: Path):
    """
    Set up logging configuration.

    Args:
        results_dir: Directory to save log files
    """
    log_file = results_dir / "logs" / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    # Initialize configuration
    config = Config()

    # Create results directory
    results_dir = config.get_results_dir()
    setup_logging(results_dir)

    logging.info("Starting baby cry detection training")
    logging.info(f"Results will be saved to: {results_dir}")

    try:
        # Initialize trainer
        trainer = BabyCryTrainer(config)

        # Setup training
        trainer.setup_training()

        # Train model
        history = trainer.train(results_dir)

        # Save inference model
        trainer.save_model_for_inference(results_dir)

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()