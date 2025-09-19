"""
Training script for OSAS transformer model.

This module provides:
- Training loop with PyTorch Lightning or custom implementation
- Support for early stopping, learning rate scheduling, class weights
- Logging of accuracy, macro-F1, per-class recall
- Model checkpointing and evaluation
"""

import os
import json
import argparse
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import wandb
from tqdm import tqdm

# Import project modules
from model import create_model, OSASTransformer, OSASMultiTaskTransformer
from dataset import OSASDataset, PatientAwareSplitter, create_data_loaders, collate_fn
from preprocess_data import OSASPreprocessor


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


class OSASTrainer:
    """Trainer class for OSAS detection models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: Dict,
                 device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model: The neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize training components
        self._setup_training()
        
    def _setup_training(self):
        """Setup optimizer, loss function, scheduler, etc."""
        # Optimizer
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function
        task = self.config.get('task', 'binary')
        use_class_weights = self.config.get('use_class_weights', True)
        
        if use_class_weights and hasattr(self.train_loader.dataset, 'get_class_weights'):
            class_weights = self.train_loader.dataset.get_class_weights().to(self.device)
            print(f"Using class weights: {class_weights}")
        else:
            class_weights = None
        
        if task == 'binary':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif task == 'multiclass':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif task == 'multitask':
            # For multi-task, we'll use separate losses
            binary_weights = class_weights[:2] if class_weights is not None else None
            multiclass_weights = class_weights if class_weights is not None and len(class_weights) == 5 else None
            
            self.binary_criterion = nn.CrossEntropyLoss(weight=binary_weights)
            self.multiclass_criterion = nn.CrossEntropyLoss(weight=multiclass_weights)
        
        # Learning rate scheduler
        scheduler_name = self.config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100),
                eta_min=lr * 0.01
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step_size', 30),
                gamma=self.config.get('scheduler_gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config.get('scheduler_patience', 10),
                factor=self.config.get('scheduler_factor', 0.5)
            )
        else:
            self.scheduler = None
        
        # Early stopping
        if self.config.get('early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=self.config.get('early_stopping_patience', 15),
                min_delta=self.config.get('early_stopping_min_delta', 0.001)
            )
        else:
            self.early_stopping = None
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Wandb logging
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'osas-detection'),
                config=self.config,
                name=self.config.get('experiment_name')
            )
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, task: str = 'binary') -> Dict:
        """Compute evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # F1 scores
        if task == 'binary':
            metrics['f1'] = f1_score(targets, predictions, average='binary')
            metrics['f1_macro'] = f1_score(targets, predictions, average='macro')
        else:
            metrics['f1_macro'] = f1_score(targets, predictions, average='macro')
            metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        for i in range(len(precision)):
            metrics[f'precision_class_{i}'] = precision[i]
            metrics[f'recall_class_{i}'] = recall[i]
            metrics[f'f1_class_{i}'] = f1[i]
            metrics[f'support_class_{i}'] = support[i]
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions).tolist()
        
        return metrics
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # For multi-task
        all_binary_predictions = []
        all_binary_targets = []
        all_multiclass_predictions = []
        all_multiclass_targets = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (features, targets) in enumerate(progress_bar):
            # Move to device
            for key in features:
                features[key] = features[key].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Compute loss
            if self.config.get('task') == 'multitask':
                # Multi-task loss
                binary_targets = (targets > 0).long()  # Convert to binary
                multiclass_targets = targets
                
                binary_loss = self.binary_criterion(outputs['binary_logits'], binary_targets)
                multiclass_loss = self.multiclass_criterion(outputs['multiclass_logits'], multiclass_targets)
                
                # Weighted combination
                binary_weight = self.config.get('binary_loss_weight', 0.5)
                multiclass_weight = self.config.get('multiclass_loss_weight', 0.5)
                loss = binary_weight * binary_loss + multiclass_weight * multiclass_loss
                
                # Collect predictions
                binary_preds = torch.argmax(outputs['binary_logits'], dim=1).cpu().numpy()
                multiclass_preds = torch.argmax(outputs['multiclass_logits'], dim=1).cpu().numpy()
                
                all_binary_predictions.extend(binary_preds)
                all_binary_targets.extend(binary_targets.cpu().numpy())
                all_multiclass_predictions.extend(multiclass_preds)
                all_multiclass_targets.extend(multiclass_targets.cpu().numpy())
            
            else:
                # Single task loss
                loss = self.criterion(outputs['logits'], targets)
                
                # Collect predictions
                predictions = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        
        if self.config.get('task') == 'multitask':
            binary_metrics = self.compute_metrics(
                np.array(all_binary_predictions), 
                np.array(all_binary_targets), 
                'binary'
            )
            multiclass_metrics = self.compute_metrics(
                np.array(all_multiclass_predictions), 
                np.array(all_multiclass_targets), 
                'multiclass'
            )
            
            metrics = {
                'loss': avg_loss,
                'binary_accuracy': binary_metrics['accuracy'],
                'binary_f1': binary_metrics['f1'],
                'multiclass_accuracy': multiclass_metrics['accuracy'],
                'multiclass_f1_macro': multiclass_metrics['f1_macro']
            }
        else:
            task_metrics = self.compute_metrics(
                np.array(all_predictions), 
                np.array(all_targets), 
                self.config.get('task', 'binary')
            )
            metrics = {'loss': avg_loss, **task_metrics}
        
        return metrics
    
    def validate(self) -> Dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # For multi-task
        all_binary_predictions = []
        all_binary_targets = []
        all_multiclass_predictions = []
        all_multiclass_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                for key in features:
                    features[key] = features[key].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Compute loss
                if self.config.get('task') == 'multitask':
                    binary_targets = (targets > 0).long()
                    multiclass_targets = targets
                    
                    binary_loss = self.binary_criterion(outputs['binary_logits'], binary_targets)
                    multiclass_loss = self.multiclass_criterion(outputs['multiclass_logits'], multiclass_targets)
                    
                    binary_weight = self.config.get('binary_loss_weight', 0.5)
                    multiclass_weight = self.config.get('multiclass_loss_weight', 0.5)
                    loss = binary_weight * binary_loss + multiclass_weight * multiclass_loss
                    
                    # Collect predictions
                    binary_preds = torch.argmax(outputs['binary_logits'], dim=1).cpu().numpy()
                    multiclass_preds = torch.argmax(outputs['multiclass_logits'], dim=1).cpu().numpy()
                    
                    all_binary_predictions.extend(binary_preds)
                    all_binary_targets.extend(binary_targets.cpu().numpy())
                    all_multiclass_predictions.extend(multiclass_preds)
                    all_multiclass_targets.extend(multiclass_targets.cpu().numpy())
                
                else:
                    loss = self.criterion(outputs['logits'], targets)
                    
                    # Collect predictions
                    predictions = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
                    all_predictions.extend(predictions)
                    all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        
        if self.config.get('task') == 'multitask':
            binary_metrics = self.compute_metrics(
                np.array(all_binary_predictions), 
                np.array(all_binary_targets), 
                'binary'
            )
            multiclass_metrics = self.compute_metrics(
                np.array(all_multiclass_predictions), 
                np.array(all_multiclass_targets), 
                'multiclass'
            )
            
            metrics = {
                'loss': avg_loss,
                'binary_accuracy': binary_metrics['accuracy'],
                'binary_f1': binary_metrics['f1'],
                'multiclass_accuracy': multiclass_metrics['accuracy'],
                'multiclass_f1_macro': multiclass_metrics['f1_macro'],
                'binary_confusion_matrix': binary_metrics['confusion_matrix'],
                'multiclass_confusion_matrix': multiclass_metrics['confusion_matrix']
            }
        else:
            task_metrics = self.compute_metrics(
                np.array(all_predictions), 
                np.array(all_targets), 
                self.config.get('task', 'binary')
            )
            metrics = {'loss': avg_loss, **task_metrics}
        
        return metrics
    
    def train(self) -> Dict:
        """Full training loop."""
        epochs = self.config.get('epochs', 100)
        save_dir = self.config.get('save_dir', './checkpoints')
        
        # Create separate directories for models and results
        models_dir = os.path.join(save_dir, 'models')
        results_dir = os.path.join(save_dir, 'results')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        best_metric = -np.inf
        best_epoch = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            if self.config.get('task') == 'multitask':
                print(f"Train Binary Acc: {train_metrics['binary_accuracy']:.4f}, F1: {train_metrics['binary_f1']:.4f}")
                print(f"Train Multiclass Acc: {train_metrics['multiclass_accuracy']:.4f}, F1: {train_metrics['multiclass_f1_macro']:.4f}")
                print(f"Val Binary Acc: {val_metrics['binary_accuracy']:.4f}, F1: {val_metrics['binary_f1']:.4f}")
                print(f"Val Multiclass Acc: {val_metrics['multiclass_accuracy']:.4f}, F1: {val_metrics['multiclass_f1_macro']:.4f}")
                
                # Use combined metric for early stopping
                combined_metric = (val_metrics['binary_f1'] + val_metrics['multiclass_f1_macro']) / 2
            else:
                if 'f1' in train_metrics:
                    print(f"Train Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
                    print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
                    combined_metric = val_metrics['f1']
                else:
                    print(f"Train Acc: {train_metrics['accuracy']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}")
                    print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}")
                    combined_metric = val_metrics['f1_macro']
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(combined_metric)
                else:
                    self.scheduler.step()
            
            # Save best model
            if combined_metric > best_metric:
                best_metric = combined_metric
                best_epoch = epoch
                
                # Include task type in model filename and save to models directory
                task_type = self.config.get('task', 'binary')
                model_filename = f'best_model_{task_type}.pth'
                model_path = os.path.join(models_dir, model_filename)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': best_metric,
                    'config': self.config
                }, model_path)
                print(f"Saved best {task_type} model with metric: {best_metric:.4f}")
                print(f"Model saved to: {model_path}")
            
            # Wandb logging
            if self.config.get('use_wandb', False):
                log_dict = {
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    **{f'train_{k}': v for k, v in train_metrics.items() if k != 'confusion_matrix'},
                    **{f'val_{k}': v for k, v in val_metrics.items() if k not in ['confusion_matrix', 'binary_confusion_matrix', 'multiclass_confusion_matrix']}
                }
                wandb.log(log_dict)
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(combined_metric, self.model):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best epoch: {best_epoch + 1} with metric: {best_metric:.4f}")
                    break
        
        # Save final metrics
        results = {
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        task_type = self.config.get('task', 'binary')
        results_filename = f'training_results_{task_type}.json'
        results_path = os.path.join(results_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Training results saved to: {results_path}")
        
        return results
    
    def test(self, model_path: Optional[str] = None) -> Dict:
        """Test the model."""
        if model_path is not None:
            # Load best model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        return self.validate()  # Use validation function on test data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train OSAS detection model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/processed_windows.pkl',
                       help='Path to processed data')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass', 'multitask'], 
                       default='binary', help='Task type')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    print("Loading data...")
    try:
        windows, metadata = OSASPreprocessor.load_processed_data(args.data_path)
        print(f"Loaded {len(windows)} windows")
    except FileNotFoundError:
        print(f"Data file not found: {args.data_path}")
        print("Please run preprocess_data.py first to generate processed data.")
        return
    
    # Create dataset
    dataset = OSASDataset(windows, task=args.task)
    
    # Split patients
    splitter = PatientAwareSplitter()
    train_patients, val_patients, test_patients = splitter.split_patients(
        dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Create split datasets
    train_dataset, val_dataset, test_dataset = splitter.create_patient_datasets(
        dataset, train_patients, val_patients, test_patients
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} windows from {len(train_patients)} patients")
    print(f"  Val: {len(val_dataset)} windows from {len(val_patients)} patients") 
    print(f"  Test: {len(test_dataset)} windows from {len(test_patients)} patients")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=args.batch_size, num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Update config with data shapes
    sample_features, _ = train_dataset[0]
    if 'vital_signs' in sample_features:
        config['vital_dim'] = sample_features['vital_signs'].shape[-1]
    if 'waveforms' in sample_features:
        config['waveform_channels'] = sample_features['waveforms'].shape[-1]
        config['waveform_length'] = sample_features['waveforms'].shape[1]
    
    # Set number of classes
    if args.task == 'binary':
        config['num_classes'] = 2
    elif args.task == 'multiclass':
        config['num_classes'] = 5
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create trainer
    trainer = OSASTrainer(model, train_loader, val_loader, test_loader, config, args.device)
    
    # Train model
    print("Starting training...")
    results = trainer.train()
    
    # Test model
    print("\nTesting best model...")
    task_type = args.task
    model_filename = f'best_model_{task_type}.pth'
    models_dir = os.path.join(args.save_dir, 'models')
    model_path = os.path.join(models_dir, model_filename)
    test_metrics = trainer.test(model_path)
    
    print("\nTest Results:")
    if args.task == 'multitask':
        print(f"Binary - Accuracy: {test_metrics['binary_accuracy']:.4f}, F1: {test_metrics['binary_f1']:.4f}")
        print(f"Multiclass - Accuracy: {test_metrics['multiclass_accuracy']:.4f}, F1: {test_metrics['multiclass_f1_macro']:.4f}")
    else:
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        if 'f1' in test_metrics:
            print(f"F1: {test_metrics['f1']:.4f}")
        if 'f1_macro' in test_metrics:
            print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    
    print(f"\nTraining completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()