"""
PyTorch Dataset class for OSAS detection.

This module provides:
- PyTorch Dataset class for loading processed windows
- Patient-aware train/val/test splitting
- Data loading utilities
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import pickle
from collections import Counter
import warnings


class OSASDataset(Dataset):
    """PyTorch Dataset for OSAS detection."""
    
    def __init__(self, 
                 windows: List[Dict],
                 task: str = 'binary',
                 transform: Optional[callable] = None):
        """
        Initialize OSAS Dataset.
        
        Args:
            windows: List of preprocessed window dictionaries
            task: 'binary' for anomaly detection, 'multiclass' for event classification
            transform: Optional transform to apply to data
        """
        self.windows = windows
        self.task = task
        self.transform = transform
        
        # Define label mappings
        self.binary_labels = {False: 0, True: 1}
        self.multiclass_labels = {
            'NONE': 0,
            'HYPOPNEA': 1,
            'APNEA-OBSTRUCTIVE': 2,
            'APNEA-MIXED': 3,
            'APNEA-CENTRAL': 4
        }
        self.inverse_multiclass_labels = {v: k for k, v in self.multiclass_labels.items()}
        
        # Validate windows and extract statistics
        self._validate_windows()
        self._compute_statistics()
    
    def _validate_windows(self):
        """Validate window data and filter invalid entries."""
        valid_windows = []
        
        for window in self.windows:
            # Check required fields
            if self.task == 'binary' and 'binary_label' not in window:
                continue
            if self.task == 'multiclass' and 'multiclass_label' not in window:
                continue
            
            # Check if window has required features
            has_vital_signs = 'vital_signs' in window and window['vital_signs'] is not None
            has_waveforms = 'waveforms' in window and window['waveforms'] is not None
            has_psg = 'psg_signals' in window and window['psg_signals'] is not None
            
            if has_vital_signs or has_waveforms or has_psg:
                valid_windows.append(window)
        
        self.windows = valid_windows
        print(f"Dataset validation: {len(self.windows)} valid windows")
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        self.n_samples = len(self.windows)
        self.patients = list(set([w['patient'] for w in self.windows]))
        self.n_patients = len(self.patients)
        
        # Label distribution
        if self.task == 'binary':
            labels = [w['binary_label'] for w in self.windows]
            self.label_distribution = Counter(labels)
        else:
            labels = [w['multiclass_label'] for w in self.windows]
            self.label_distribution = Counter(labels)
        
        # Feature shapes
        sample_window = self.windows[0]
        self.feature_shapes = {}
        if 'vital_signs' in sample_window:
            self.feature_shapes['vital_signs'] = sample_window['vital_signs'].shape
        if 'waveforms' in sample_window:
            self.feature_shapes['waveforms'] = sample_window['waveforms'].shape
        if 'psg_signals' in sample_window:
            self.feature_shapes['psg_signals'] = sample_window['psg_signals'].shape
        
        print(f"Dataset statistics:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Patients: {self.n_patients}")
        print(f"  Label distribution: {dict(self.label_distribution)}")
        print(f"  Feature shapes: {self.feature_shapes}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (features_dict, label)
        """
        window = self.windows[idx]
        
        # Extract features
        features = {}
        
        # Vital signs
        if 'vital_signs' in window and window['vital_signs'] is not None:
            vital_signs = torch.FloatTensor(window['vital_signs'])
            features['vital_signs'] = vital_signs
        
        # Waveforms (ECG, PPG)
        if 'waveforms' in window and window['waveforms'] is not None:
            waveforms = torch.FloatTensor(window['waveforms'])
            features['waveforms'] = waveforms
        
        # PSG signals
        if 'psg_signals' in window and window['psg_signals'] is not None:
            psg_signals = torch.FloatTensor(window['psg_signals'])
            features['psg_signals'] = psg_signals
        
        # Extract label
        if self.task == 'binary':
            label = self.binary_labels[window['binary_label']]
        else:
            label = self.multiclass_labels.get(window['multiclass_label'], 0)
        
        label = torch.LongTensor([label])[0]  # Convert to scalar tensor
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced data."""
        if self.task == 'binary':
            n_classes = 2
            class_counts = [self.label_distribution.get(i, 0) for i in [False, True]]
        else:
            n_classes = len(self.multiclass_labels)
            class_counts = []
            for label in self.multiclass_labels.keys():
                class_counts.append(self.label_distribution.get(label, 0))
        
        # Compute inverse frequency weights
        total_samples = sum(class_counts)
        weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (n_classes * count)
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_patient_data(self, patient_id: int) -> List[Dict]:
        """Get all windows for a specific patient."""
        return [w for w in self.windows if w['patient'] == patient_id]


class PatientAwareSplitter:
    """Utility class for patient-aware data splitting."""
    
    @staticmethod
    def split_patients(dataset: OSASDataset, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_state: int = 42,
                      stratify_by_label: bool = True) -> Tuple[List[int], List[int], List[int]]:
        """
        Split patients into train/val/test sets.
        
        Args:
            dataset: OSASDataset instance
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_state: Random seed
            stratify_by_label: Whether to stratify by patient-level label distribution
            
        Returns:
            Tuple of (train_patients, val_patients, test_patients)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, val, and test ratios must sum to 1.0"
        
        patients = dataset.patients
        
        if stratify_by_label:
            # Calculate patient-level label distributions
            patient_labels = {}
            for patient in patients:
                patient_windows = dataset.get_patient_data(patient)
                if dataset.task == 'binary':
                    # Use anomaly rate as stratification criterion
                    anomaly_rate = sum(w['binary_label'] for w in patient_windows) / len(patient_windows)
                    # Discretize into high/low anomaly rate
                    patient_labels[patient] = 1 if anomaly_rate > 0.2 else 0
                else:
                    # Use most frequent event type
                    events = [w['multiclass_label'] for w in patient_windows]
                    most_frequent = Counter(events).most_common(1)[0][0]
                    patient_labels[patient] = dataset.multiclass_labels.get(most_frequent, 0)
            
            # Stratified split
            patients_array = np.array(patients)
            labels_array = np.array([patient_labels[p] for p in patients])
            
            # First split: train vs (val + test)
            train_patients, temp_patients, _, temp_labels = train_test_split(
                patients_array, labels_array,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=labels_array
            )
            
            # Second split: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=(1 - val_size),
                random_state=random_state,
                stratify=temp_labels
            )
        
        else:
            # Simple random split
            patients_array = np.array(patients)
            
            # First split: train vs (val + test)
            train_patients, temp_patients = train_test_split(
                patients_array,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )
            
            # Second split: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=(1 - val_size),
                random_state=random_state
            )
        
        return train_patients.tolist(), val_patients.tolist(), test_patients.tolist()
    
    @staticmethod
    def create_patient_datasets(dataset: OSASDataset,
                               train_patients: List[int],
                               val_patients: List[int],
                               test_patients: List[int]) -> Tuple[OSASDataset, OSASDataset, OSASDataset]:
        """
        Create separate datasets for train/val/test based on patient splits.
        
        Args:
            dataset: Original dataset
            train_patients: List of patient IDs for training
            val_patients: List of patient IDs for validation
            test_patients: List of patient IDs for testing
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_windows = []
        val_windows = []
        test_windows = []
        
        for window in dataset.windows:
            patient_id = window['patient']
            if patient_id in train_patients:
                train_windows.append(window)
            elif patient_id in val_patients:
                val_windows.append(window)
            elif patient_id in test_patients:
                test_windows.append(window)
        
        train_dataset = OSASDataset(train_windows, task=dataset.task, transform=dataset.transform)
        val_dataset = OSASDataset(val_windows, task=dataset.task, transform=dataset.transform)
        test_dataset = OSASDataset(test_windows, task=dataset.task, transform=dataset.transform)
        
        return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset: OSASDataset,
                       val_dataset: OSASDataset,
                       test_dataset: OSASDataset,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Custom collate function for handling variable-sized features.
    
    Args:
        batch: List of (features, label) tuples
        
    Returns:
        Tuple of (batched_features, batched_labels)
    """
    features_list, labels_list = zip(*batch)
    
    # Batch labels
    labels = torch.stack(labels_list)
    
    # Batch features
    batched_features = {}
    
    # Find common feature keys
    feature_keys = set(features_list[0].keys())
    for features in features_list[1:]:
        feature_keys = feature_keys.intersection(set(features.keys()))
    
    # Batch each feature type
    for key in feature_keys:
        feature_tensors = [features[key] for features in features_list]
        try:
            batched_features[key] = torch.stack(feature_tensors)
        except RuntimeError:
            # Handle size mismatches by padding
            max_shape = feature_tensors[0].shape
            for tensor in feature_tensors[1:]:
                max_shape = tuple(max(a, b) for a, b in zip(max_shape, tensor.shape))
            
            padded_tensors = []
            for tensor in feature_tensors:
                pad_width = []
                for i in range(len(max_shape)):
                    pad_width.extend([0, max_shape[i] - tensor.shape[i]])
                pad_width.reverse()
                padded = torch.nn.functional.pad(tensor, pad_width)
                padded_tensors.append(padded)
            
            batched_features[key] = torch.stack(padded_tensors)
    
    return batched_features, labels


def main():
    """Example usage of the dataset classes."""
    # Load processed data
    try:
        from preprocess_data import OSASPreprocessor
        
        windows, metadata = OSASPreprocessor.load_processed_data('./data/processed_windows.pkl')
        print(f"Loaded {len(windows)} processed windows")
        
        # Create datasets
        binary_dataset = OSASDataset(windows, task='binary')
        multiclass_dataset = OSASDataset(windows, task='multiclass')
        
        # Split patients
        splitter = PatientAwareSplitter()
        
        # Binary classification split
        train_patients, val_patients, test_patients = splitter.split_patients(
            binary_dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15,
            stratify_by_label=True
        )
        
        print(f"Patient split:")
        print(f"  Train: {len(train_patients)} patients")
        print(f"  Validation: {len(val_patients)} patients")
        print(f"  Test: {len(test_patients)} patients")
        
        # Create split datasets
        train_dataset, val_dataset, test_dataset = splitter.create_patient_datasets(
            binary_dataset, train_patients, val_patients, test_patients
        )
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(train_dataset)} windows")
        print(f"  Validation: {len(val_dataset)} windows")
        print(f"  Test: {len(test_dataset)} windows")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=16
        )
        
        # Test data loading
        print(f"Testing data loading...")
        for batch_features, batch_labels in train_loader:
            print(f"Batch features: {list(batch_features.keys())}")
            print(f"Batch labels shape: {batch_labels.shape}")
            for key, tensor in batch_features.items():
                print(f"  {key} shape: {tensor.shape}")
            break
        
        # Print class weights
        class_weights = binary_dataset.get_class_weights()
        print(f"Class weights: {class_weights}")
        
    except FileNotFoundError:
        print("Processed data not found. Please run preprocess_data.py first.")


if __name__ == "__main__":
    main()
