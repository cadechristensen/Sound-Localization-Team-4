"""
Evaluation module for baby cry detection model.
Implements comprehensive model evaluation with various metrics and visualizations.
"""

import sys
import io

# Fix encoding issues on Windows
if sys.platform == 'win32':
    # Ensure stdout/stderr use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Use non-interactive matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd

from .config import Config
from .model import create_model
from .dataset import DatasetManager


def predict_with_tta(model: nn.Module, spectrograms: torch.Tensor, device: torch.device, n_augments: int = 5) -> torch.Tensor:
    """
    Test-time augmentation for more robust predictions.

    Args:
        model: Trained model
        spectrograms: Input spectrograms (batch_size, 1, n_mels, time_steps)
        device: torch device
        n_augments: Number of augmented versions to average

    Returns:
        Averaged predictions (batch_size, num_classes)
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original prediction
        outputs = model(spectrograms)
        predictions.append(outputs)

        # Augmented predictions
        for _ in range(n_augments - 1):
            # Time shift (shift spectrogram along time axis)
            shift_amount = torch.randint(-5, 6, (1,)).item()
            aug_spec = torch.roll(spectrograms, shifts=shift_amount, dims=-1)

            # Add slight noise
            noise = torch.randn_like(aug_spec) * 0.01
            aug_spec = aug_spec + noise

            outputs = model(aug_spec)
            predictions.append(outputs)

    # Average predictions
    avg_outputs = torch.mean(torch.stack(predictions), dim=0)
    return avg_outputs


class EnsembleModel:
    """Ensemble multiple model checkpoints for better predictions."""

    def __init__(self, model_paths: List[Path], config: Config, device: torch.device):
        """
        Initialize ensemble.

        Args:
            model_paths: List of paths to model checkpoints
            config: Config object
            device: torch device
        """
        self.models = []
        self.device = device
        self.config = config

        logging.info(f"Loading {len(model_paths)} models for ensemble...")
        for i, path in enumerate(model_paths):
            if not path.exists():
                logging.warning(f"Model path not found: {path}, skipping...")
                continue

            # Always load to CPU first, then move to target device
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

            # Standard format: load state_dict into a fresh model
            model = create_model(config).to(device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self.models.append(model)
            logging.info(f"  Loaded model {i+1}/{len(model_paths)}: {path.name}")

        if not self.models:
            raise ValueError("No valid models loaded for ensemble!")

        logging.info(f"Ensemble ready with {len(self.models)} models")

    def eval(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all models and average predictions."""
        with torch.no_grad():
            predictions = [model(x) for model in self.models]
            return torch.mean(torch.stack(predictions), dim=0)


class ModelEvaluator:
    """
    Comprehensive model evaluator for baby cry detection.
    Provides various metrics and visualizations for model performance analysis.
    """

    def __init__(self, config: Config = Config(), use_tta: bool = False, tta_n_augments: int = 5):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object
            use_tta: Whether to use test-time augmentation
            tta_n_augments: Number of augmentations for TTA
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = list(config.CLASS_LABELS.values())
        self.use_tta = use_tta
        self.tta_n_augments = tta_n_augments

        # Initialize model
        self.model = create_model(config).to(self.device)

    @staticmethod
    def _setup_module_aliases():
        """Setup module aliases to handle different import paths when unpickling models."""
        import sys
        from . import model, config

        # Ensure 'src' module exists as an alias to the current package
        if 'src' not in sys.modules:
            sys.modules['src'] = sys.modules[__package__]

        # Ensure individual modules are accessible both ways
        if 'src.model' not in sys.modules:
            sys.modules['src.model'] = model
        if 'src.config' not in sys.modules:
            sys.modules['src.config'] = config

        # Also make them available without 'src.' prefix (for standalone imports)
        if 'model' not in sys.modules:
            sys.modules['model'] = model
        if 'config' not in sys.modules:
            sys.modules['config'] = config

    def load_model(self, checkpoint_path: Path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Setup module aliases for loading pickled models
        # This allows loading models that were saved with different module paths
        self._setup_module_aliases()

        # Always load to CPU first, then move to target device
        # This handles models saved on CUDA when CUDA is not available
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load model state - handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Standard checkpoint format (best model, inference model, etc.)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        else:
            # Fallback: assume the entire checkpoint is a state_dict
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)

        self.model.eval()
        logging.info(f"Model loaded from {checkpoint_path}")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            desc = "Generating predictions (TTA)" if self.use_tta else "Generating predictions"
            for batch_data in tqdm(data_loader, desc=desc):
                # Handle both old format (specs, labels) and new format (specs, labels, indices)
                if len(batch_data) == 3:
                    spectrograms, labels, _ = batch_data
                else:
                    spectrograms, labels = batch_data

                # Move to device
                spectrograms = spectrograms.to(self.device, non_blocking=True)
                labels = labels.cpu().numpy()

                # Forward pass with optional TTA
                if self.use_tta:
                    outputs = predict_with_tta(self.model, spectrograms, self.device, self.tta_n_augments)
                else:
                    outputs = self.model(spectrograms)

                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(labels)

        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            np.array(all_labels)
        )

    def predict_and_log_errors(self, data_loader: DataLoader, dataset_name: str, results_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions and log misclassified files.

        Args:
            data_loader: DataLoader for the dataset
            dataset_name: Name of dataset (train/val/test)
            results_dir: Directory to save error log

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        misclassified_files = []

        # Check if dataloader uses a sampler (which shuffles the order)
        uses_sampler = data_loader.sampler is not None

        with torch.no_grad():
            desc = "Generating predictions (TTA)" if self.use_tta else "Generating predictions"
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=desc)):
                # Handle both old format (specs, labels) and new format (specs, labels, indices)
                if len(batch_data) == 3:
                    spectrograms, labels, indices = batch_data
                    indices = indices.cpu().numpy()
                    has_indices = True
                else:
                    spectrograms, labels = batch_data
                    indices = None
                    has_indices = False

                # Move to device
                spectrograms = spectrograms.to(self.device, non_blocking=True)
                labels = labels.cpu().numpy()

                # Forward pass with optional TTA
                if self.use_tta:
                    outputs = predict_with_tta(self.model, spectrograms, self.device, self.tta_n_augments)
                else:
                    outputs = self.model(spectrograms)

                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                # Check for misclassifications
                for i, (pred, true_label, prob) in enumerate(zip(predictions, labels, probabilities)):
                    # Get filename from dataset using indices from dataloader
                    filename = "unknown"
                    if hasattr(data_loader.dataset, 'audio_files') and has_indices:
                        try:
                            # Use the actual index from the dataloader
                            dataset_idx = indices[i]
                            if 0 <= dataset_idx < len(data_loader.dataset.audio_files):
                                file_path, _ = data_loader.dataset.audio_files[dataset_idx]
                                filename = str(file_path)
                        except Exception as e:
                            logging.debug(f"Could not retrieve filename using index {indices[i]}: {e}")
                    elif hasattr(data_loader.dataset, 'audio_files') and not uses_sampler:
                        try:
                            # Fallback: For non-sampled dataloaders (val/test), try sequential indexing
                            sample_idx = batch_idx * data_loader.batch_size + i
                            if sample_idx < len(data_loader.dataset.audio_files):
                                file_path, _ = data_loader.dataset.audio_files[sample_idx]
                                filename = str(file_path)
                        except Exception as e:
                            logging.debug(f"Could not retrieve filename for batch {batch_idx}, item {i}: {e}")

                    # Log misclassifications
                    if pred != true_label:
                        # Always use numeric labels for consistency
                        true_label_name = self.class_labels[true_label]
                        pred_label_name = self.class_labels[pred]
                        confidence = prob[pred] * 100

                        misclassified_files.append({
                            'filename': filename,
                            'true_label': true_label_name,
                            'predicted_label': pred_label_name,
                            'confidence': float(confidence),
                            'probabilities': [float(p) for p in prob]
                        })

                        logging.error(f"MISCLASSIFIED - File: {filename}, True: {true_label_name}, Predicted: {pred_label_name} ({confidence:.1f}% confidence)")

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(labels)

        # Save misclassified files to JSON
        if misclassified_files:
            error_log_path = results_dir / f"misclassified_{dataset_name}.json"
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump(misclassified_files, f, indent=2, ensure_ascii=False)

            logging.info(f"Saved {len(misclassified_files)} misclassified files to {error_log_path}")

            # Also create a summary text file
            summary_path = results_dir / f"misclassified_{dataset_name}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"MISCLASSIFIED FILES SUMMARY - {dataset_name.upper()} SET\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total misclassifications: {len(misclassified_files)}\n\n")

                for item in misclassified_files:
                    f.write(f"File: {item['filename']}\n")
                    f.write(f"  True Label: {item['true_label']}\n")
                    f.write(f"  Predicted: {item['predicted_label']} ({item['confidence']:.1f}% confidence)\n")
                    f.write(f"  Probabilities: non_cry={item['probabilities'][0]:.3f}, cry={item['probabilities'][1]:.3f}\n")
                    f.write("-" * 40 + "\n")

            logging.info(f"Summary saved to {summary_path}")
        else:
            logging.info(f"No misclassifications found in {dataset_name} set!")

        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            np.array(all_labels)
        )

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary containing all metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        for i, label in enumerate(self.class_labels):
            metrics[f'precision_{label}'] = precision_per_class[i]
            metrics[f'recall_{label}'] = recall_per_class[i]
            metrics[f'f1_score_{label}'] = f1_per_class[i]

        # ROC AUC and PR metrics
        if len(self.class_labels) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            metrics['roc_auc'] = auc(fpr, tpr)

            # Precision-Recall AUC
            precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_proba[:, 1])
            metrics['pr_auc'] = auc(recall_pr, precision_pr)
            metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
        else:
            # Multi-class: compute macro-averaged metrics
            try:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(len(self.class_labels)))
                # Macro-averaged ROC AUC
                metrics['roc_auc'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                # Macro-averaged precision
                metrics['average_precision'] = average_precision_score(y_true_bin, y_proba, average='macro')
            except Exception as e:
                logging.warning(f"Could not calculate multi-class ROC/PR metrics: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_labels, output_dict=True
        )

        return metrics

    def plot_confusion_matrix(self, cm: np.ndarray, results_dir: Path):
        """
        Plot and save enhanced confusion matrix with detailed analysis.

        Args:
            cm: Confusion matrix
            results_dir: Directory to save the plot
        """
        # Create a figure with subplots for multiple visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Define clear labels based on number of classes
        if cm.shape[0] == 2:
            # Binary classification
            display_labels = ['Non-Cry', 'Cry']
        else:
            # Multi-class: use actual class labels
            display_labels = [label.replace('_', ' ').title() for label in self.class_labels]

        # 1. Raw count confusion matrix
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax1,
            cbar_kws={'label': 'Count'}
        )
        ax1.set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)

        # Add performance metrics on the plot (works for both binary and multi-class)
        if cm.shape == (2, 2):
            # Binary classification
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            ax1.text(0.5, -0.15, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}',
                    horizontalalignment='center', transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            # Multi-class: just show overall accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            ax1.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.3f}',
                    horizontalalignment='center', transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # 2. Normalized confusion matrix (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax2,
            cbar_kws={'label': 'Percentage'}
        )
        ax2.set_title('Confusion Matrix - Normalized (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)

        # 3. Detailed breakdown (binary only) or per-class metrics (multi-class)
        if cm.shape == (2, 2):
            # Binary: Detailed breakdown with labels
            labels = ['True Negative\n(Correct Non-Cry)', 'False Positive\n(Wrong: Predicted Cry)',
                     'False Negative\n(Wrong: Predicted Non-Cry)', 'True Positive\n(Correct Cry)']
            values = [tn, fp, fn, tp]
            colors = ['lightgreen', 'lightcoral', 'lightcoral', 'lightgreen']

            # Create custom confusion matrix with detailed labels
            detailed_cm = np.array([[tn, fp], [fn, tp]])
            mask = np.zeros_like(detailed_cm, dtype=bool)

            # Custom annotations
            annot_array = np.array([[f'{tn}\n{labels[0]}', f'{fp}\n{labels[1]}'],
                                   [f'{fn}\n{labels[2]}', f'{tp}\n{labels[3]}']])

            sns.heatmap(
                detailed_cm, annot=annot_array, fmt='',
                xticklabels=display_labels,
                yticklabels=display_labels,
                ax=ax3, cmap='RdYlGn', center=max(tn, fp, fn, tp)/2,
                cbar_kws={'label': 'Count'}
            )
            ax3.set_title('Confusion Matrix - Detailed Analysis', fontsize=14, fontweight='bold')
        else:
            # Multi-class: Show per-class recall (sensitivity)
            per_class_recall = cm.diagonal() / cm.sum(axis=1)
            x_pos = np.arange(len(display_labels))
            bars = ax3.bar(x_pos, per_class_recall, color='skyblue')
            ax3.set_title('Per-Class Recall (Sensitivity)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Recall', fontsize=12)
            ax3.set_ylim(0, 1.05)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(display_labels, rotation=45, ha='right')
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{per_class_recall[i]:.2f}',
                        ha='center', va='bottom', fontsize=9)

        ax3.set_xlabel('Class', fontsize=12) if cm.shape != (2, 2) else ax3.set_xlabel('Predicted Label', fontsize=12)
        ax3.set_ylabel('True Label', fontsize=12) if cm.shape == (2, 2) else None

        # 4. Performance metrics visualization
        if cm.shape == (2, 2):
            metrics_data = {
                'Accuracy': accuracy,
                'Precision (Cry)': precision,
                'Recall (Cry)': recall,
                'F1-Score': f1,
                'Specificity (Non-Cry)': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
        else:
            # Multi-class: show per-class precision
            per_class_precision = cm.diagonal() / cm.sum(axis=0)
            per_class_precision = np.nan_to_num(per_class_precision, nan=0.0)
            metrics_data = {display_labels[i]: per_class_precision[i] for i in range(len(display_labels))}

        # Color scheme based on number of metrics
        if len(metrics_data) == 5:
            colors_list = ['skyblue', 'lightgreen', 'orange', 'gold', 'lightcoral']
        else:
            colors_list = plt.cm.Set3(np.linspace(0, 1, len(metrics_data)))

        bars = ax4.bar(range(len(metrics_data)), list(metrics_data.values()), color=colors_list)
        ax4.set_title('Performance Metrics' if cm.shape == (2, 2) else 'Per-Class Precision', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_ylim(0, 1.05)
        ax4.set_xticks(range(len(metrics_data)))
        ax4.set_xticklabels(list(metrics_data.keys()), rotation=45, ha='right')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metrics_data.values())):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save comprehensive confusion matrix
        plot_path = results_dir / "plots" / "confusion_matrix_comprehensive.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Also create the original simple matrices for compatibility
        # Simple raw count matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Baby Cry Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # Add interpretation text (binary only)
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            plt.figtext(0.5, 0.02,
                       f'TP: {tp} (Correctly identified cries) | TN: {tn} (Correctly identified non-cries)\n'
                       f'FP: {fp} (False alarms - predicted cry but was non-cry) | FN: {fn} (Missed cries - predicted non-cry but was cry)',
                       ha='center', fontsize=10, style='italic',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            plt.subplots_adjust(bottom=0.15)

        plt.tight_layout()

        plot_path = results_dir / "plots" / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Simple normalized matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title('Normalized Confusion Matrix - Baby Cry Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        plot_path = results_dir / "plots" / "confusion_matrix_normalized.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Enhanced confusion matrices saved to {results_dir / 'plots'}")

        # Print detailed analysis to console and log
        if cm.shape[0] == 2:
            # Binary classification detailed logging
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            logging.info("=== CONFUSION MATRIX ANALYSIS (Binary) ===")
            logging.info(f"True Positives (TP): {tp} - Correctly identified baby cries")
            logging.info(f"True Negatives (TN): {tn} - Correctly identified non-cries")
            logging.info(f"False Positives (FP): {fp} - False alarms (predicted cry, but was non-cry)")
            logging.info(f"False Negatives (FN): {fn} - Missed cries (predicted non-cry, but was cry)")
            logging.info(f"Accuracy: {accuracy:.3f} - Overall correctness")
            logging.info(f"Precision: {precision:.3f} - Of all cry predictions, how many were correct")
            logging.info(f"Recall: {recall:.3f} - Of all actual cries, how many were detected")
            logging.info(f"F1-Score: {f1:.3f} - Balanced measure of precision and recall")
        else:
            # Multi-class summary logging
            accuracy = np.trace(cm) / np.sum(cm)
            logging.info("=== CONFUSION MATRIX ANALYSIS (Multi-Class) ===")
            logging.info(f"Overall Accuracy: {accuracy:.3f}")
            logging.info(f"Class-wise performance:")
            for i, label in enumerate(display_labels):
                class_total = cm[i].sum()
                class_correct = cm[i, i]
                class_recall = class_correct / class_total if class_total > 0 else 0
                logging.info(f"  {label}: {class_correct}/{class_total} correct (recall: {class_recall:.3f})")

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, results_dir: Path):
        """
        Plot and save ROC curve.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            results_dir: Directory to save the plot
        """
        if len(self.class_labels) != 2:
            logging.warning("ROC curve only applicable for binary classification")
            return

        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = results_dir / "plots" / "roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"ROC curve saved to {plot_path}")

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, results_dir: Path):
        """
        Plot and save Precision-Recall curve.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            results_dir: Directory to save the plot
        """
        if len(self.class_labels) != 2:
            logging.warning("PR curve only applicable for binary classification")
            return

        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y_true, y_proba[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f}, AP = {avg_precision:.3f})')

        # Random baseline
        pos_ratio = np.sum(y_true) / len(y_true)
        plt.axhline(y=pos_ratio, color='red', linestyle='--',
                   label=f'Random (AP = {pos_ratio:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = results_dir / "plots" / "precision_recall_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Precision-Recall curve saved to {plot_path}")

    def plot_training_history(self, history: Dict, results_dir: Path):
        """
        Plot training history (loss and accuracy).

        Args:
            history: Training history dictionary
            results_dir: Directory to save the plot
        """
        if not history:
            logging.warning("No training history found")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_ylim(0, 0.1)  # Custom y-axis range for loss (0 to 0.6)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0.6, 1.0)  # Custom y-axis range for accuracy (60% to 100%)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate plot
        if 'learning_rates' in history:
            ax3.plot(epochs, history['learning_rates'], 'g-', label='Learning Rate')
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = results_dir / "plots" / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Training history plot saved to {plot_path}")

    def plot_class_distribution(self, data_loader: DataLoader, results_dir: Path, dataset_name: str):
        """
        Plot class distribution in the dataset.

        Args:
            data_loader: DataLoader for the dataset
            results_dir: Directory to save the plot
            dataset_name: Name of the dataset (train/val/test)
        """
        class_counts = {label: 0 for label in self.class_labels}

        for batch_data in data_loader:
            # Handle both old format (specs, labels) and new format (specs, labels, indices)
            if len(batch_data) == 3:
                _, labels, _ = batch_data
            else:
                _, labels = batch_data

            for label in labels:
                class_name = self.class_labels[label.item()]
                class_counts[class_name] += 1

        # Create bar plot
        plt.figure(figsize=(8, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
        plt.title(f'Class Distribution - {dataset_name.title()} Set')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')

        plt.tight_layout()

        plot_path = results_dir / "plots" / f"class_distribution_{dataset_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Class distribution plot saved to {plot_path}")

    def evaluate_model(
        self,
        data_loader: DataLoader,
        results_dir: Path,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            data_loader: DataLoader for evaluation
            results_dir: Directory to save results
            dataset_name: Name of the dataset being evaluated

        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info(f"Evaluating model on {dataset_name} set...")

        # Generate predictions and log errors
        y_pred, y_proba, y_true = self.predict_and_log_errors(data_loader, dataset_name, results_dir)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)

        # Create visualizations
        cm = np.array(metrics['confusion_matrix'])
        self.plot_confusion_matrix(cm, results_dir)
        self.plot_roc_curve(y_true, y_proba, results_dir)
        self.plot_precision_recall_curve(y_true, y_proba, results_dir)
        self.plot_class_distribution(data_loader, results_dir, dataset_name)

        # Save metrics to file
        metrics_file = results_dir / f"metrics_{dataset_name}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, np.floating):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)

        # Print summary
        logging.info(f"\n{dataset_name.upper()} SET EVALUATION RESULTS:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1-Score: {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")

        return metrics


def evaluate_saved_model(model_path: Path, config: Config = Config()):
    """
    Evaluate a saved model.

    Args:
        model_path: Path to saved model
        config: Configuration object
    """
    # Create results directory
    results_dir = config.get_results_dir()

    # Setup logging
    log_file = results_dir / "logs" / "evaluation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    evaluator.load_model(model_path)

    # Prepare dataset
    dataset_manager = DatasetManager(config)
    train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
    train_loader, val_loader, test_loader = dataset_manager.create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )

    # Evaluate on all splits
    train_metrics = evaluator.evaluate_model(train_loader, results_dir, "train")
    val_metrics = evaluator.evaluate_model(val_loader, results_dir, "val")
    test_metrics = evaluator.evaluate_model(test_loader, results_dir, "test")

    # Load and plot training history if available
    history_file = model_path.parent / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        evaluator.plot_training_history(history, results_dir)

    logging.info(f"Evaluation completed. Results saved to {results_dir}")

    return train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            sys.exit(1)
    else:
        print("Usage: python evaluate.py <path_to_model>")
        sys.exit(1)

    config = Config()
    evaluate_saved_model(model_path, config)
