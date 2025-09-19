"""
Data preprocessing pipeline for OSAS detection project.

This module implements the preprocessing steps outlined in the project prompt:
1. Apply 2nd-order Butterworth bandpass filter (5-35 Hz) to ECG + PPG waveforms
2. Segment each patient's data into overlapping 60-second windows (50% overlap)
3. Assign labels to each window (binary/multiclass)
4. Drop windows with >50% missing data
5. Normalize signals per-patient
"""

import pandas as pd
import numpy as np
import pickle
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
from tqdm import tqdm
import os


class OSASPreprocessor:
    """Preprocessing pipeline for OSAS detection dataset."""
    
    def __init__(self, 
                 window_size: int = 60,
                 overlap: float = 0.5,
                 filter_order: int = 2,
                 filter_low: float = 5.0,
                 filter_high: float = 35.0,
                 missing_threshold: float = 0.5,
                 sampling_rates: Dict[str, int] = None):
        """
        Initialize preprocessor with configuration parameters.
        
        Args:
            window_size: Window size in seconds (default: 60)
            overlap: Overlap fraction between windows (default: 0.5)
            filter_order: Butterworth filter order (default: 2)
            filter_low: Low cutoff frequency in Hz (default: 5.0)
            filter_high: High cutoff frequency in Hz (default: 35.0)
            missing_threshold: Maximum fraction of missing data per window (default: 0.5)
            sampling_rates: Dictionary of sampling rates per signal type
        """
        self.window_size = window_size
        self.overlap = overlap
        self.filter_order = filter_order
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.missing_threshold = missing_threshold
        
        # Default sampling rates from prompt
        self.sampling_rates = sampling_rates or {
            'ecg': 80,
            'ppg': 80,
            'psg_flow': 20,
            'psg_snore': 10,
            'psg_position': 10,
            'psg_thorax': 10,
            'psg_abdomen': 10
        }
        
        # Signal column mappings
        self.vital_columns = ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PVCs(/min)']
        self.waveform_columns = ['signal_ecg_i', 'signal_ecg_ii', 'signal_ecg_iii', 'signal_pleth']
        self.psg_columns = ['PSG_Flow', 'PSG_Snore', 'PSG_Position', 'PSG_Thorax', 'PSG_Abdomen']
        
        # Store patient-specific scalers
        self.patient_scalers = {}
        
    def apply_bandpass_filter(self, signal_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply 2nd-order Butterworth bandpass filter to waveform data.
        
        Args:
            signal_data: Input waveform data
            sampling_rate: Sampling rate of the signal
            
        Returns:
            Filtered signal data
        """
        try:
            # Design Butterworth bandpass filter
            nyquist = sampling_rate / 2
            low_norm = self.filter_low / nyquist
            high_norm = self.filter_high / nyquist
            
            # Ensure frequencies are within valid range
            low_norm = max(0.01, min(low_norm, 0.99))
            high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
            
            b, a = signal.butter(self.filter_order, [low_norm, high_norm], btype='band')
            
            # Handle different input shapes
            if signal_data.ndim == 1:
                if len(signal_data) < 3 * self.filter_order:
                    # Signal too short for filtering
                    return signal_data
                filtered_data = signal.filtfilt(b, a, signal_data)
            else:
                # Apply filter to each channel
                filtered_data = np.zeros_like(signal_data)
                for i in range(signal_data.shape[1]):
                    channel_data = signal_data[:, i]
                    if len(channel_data) >= 3 * self.filter_order and not np.all(np.isnan(channel_data)):
                        valid_mask = ~np.isnan(channel_data)
                        if np.sum(valid_mask) >= 3 * self.filter_order:
                            filtered_data[:, i] = signal.filtfilt(b, a, channel_data)
                        else:
                            filtered_data[:, i] = channel_data
                    else:
                        filtered_data[:, i] = channel_data
                        
            return filtered_data
            
        except Exception as e:
            warnings.warn(f"Filter application failed: {e}. Returning original signal.")
            return signal_data
    
    def preprocess_waveforms(self, waveform_data: np.ndarray, signal_type: str) -> np.ndarray:
        """
        Preprocess waveform data (ECG, PPG, PSG signals).
        
        Args:
            waveform_data: Raw waveform data
            signal_type: Type of signal ('ecg', 'ppg', 'psg')
            
        Returns:
            Preprocessed waveform data
        """
        if waveform_data is None or len(waveform_data) == 0:
            return np.array([])
            
        try:
            # Convert to numpy array
            if not isinstance(waveform_data, np.ndarray):
                waveform_data = np.array(waveform_data)
            
            # Handle scalar values
            if waveform_data.ndim == 0:
                return np.array([waveform_data])
            
            # Flatten if needed
            if waveform_data.ndim > 1:
                waveform_data = waveform_data.flatten()
            
            # Get sampling rate for signal type
            if signal_type in ['ecg', 'ppg']:
                fs = self.sampling_rates.get(signal_type, 80)
                # Apply bandpass filter for ECG and PPG
                return self.apply_bandpass_filter(waveform_data, fs)
            else:
                # For PSG signals, apply less aggressive processing
                return waveform_data
                
        except Exception as e:
            warnings.warn(f"Waveform preprocessing failed for {signal_type}: {e}")
            # Return zeros if processing fails
            expected_length = {'ecg': 80, 'ppg': 80, 'psg': 10}.get(signal_type, 80)
            return np.zeros(expected_length)
    
    def calculate_missing_rate(self, data: Union[np.ndarray, float, int]) -> float:
        """Calculate missing data rate for a signal."""
        if data is None:
            return 1.0
            
        if np.isscalar(data):
            return float(np.isnan(data))
        
        try:
            data_array = np.array(data)
            if data_array.size == 0:
                return 1.0
            return np.sum(np.isnan(data_array)) / data_array.size
        except:
            return 1.0
    
    def create_windows(self, patient_data: pd.DataFrame) -> List[Dict]:
        """
        Create overlapping windows from patient data.
        
        Args:
            patient_data: DataFrame containing data for a single patient
            
        Returns:
            List of window dictionaries
        """
        windows = []
        n_samples = len(patient_data)
        step_size = int(self.window_size * (1 - self.overlap))
        
        # Sort by timestamp if available
        if 'timestamp_datetime' in patient_data.columns:
            patient_data = patient_data.sort_values('timestamp_datetime')
        
        for start_idx in range(0, n_samples - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            window_data = patient_data.iloc[start_idx:end_idx].copy()
            
            # Check missing data threshold
            total_missing_rate = 0
            signal_count = 0
            
            # Check vital signs
            for col in self.vital_columns:
                if col in window_data.columns:
                    missing_rate = window_data[col].isna().mean()
                    total_missing_rate += missing_rate
                    signal_count += 1
            
            # Check waveforms
            for col in self.waveform_columns + self.psg_columns:
                if col in window_data.columns:
                    missing_rates = [self.calculate_missing_rate(data) for data in window_data[col]]
                    missing_rate = np.mean(missing_rates)
                    total_missing_rate += missing_rate
                    signal_count += 1
            
            # Calculate average missing rate
            avg_missing_rate = total_missing_rate / signal_count if signal_count > 0 else 1.0
            
            # Skip window if too much missing data
            if avg_missing_rate > self.missing_threshold:
                continue
            
            # Assign window labels
            window_labels = self.assign_window_labels(window_data)
            
            # Create window dictionary
            window = {
                'patient': patient_data.iloc[0]['patient'],
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'window_size': self.window_size,
                'missing_rate': avg_missing_rate,
                **window_labels
            }
            
            # Add processed signals
            window.update(self.extract_window_features(window_data))
            
            windows.append(window)
        
        return windows
    
    def assign_window_labels(self, window_data: pd.DataFrame) -> Dict:
        """
        Assign labels to a window using majority voting.
        
        Args:
            window_data: DataFrame containing window data
            
        Returns:
            Dictionary with assigned labels
        """
        labels = {}
        
        # Binary classification (anomaly detection)
        if 'anomaly' in window_data.columns:
            # Use "any anomaly" strategy for binary classification
            labels['binary_label'] = window_data['anomaly'].any()
            labels['anomaly_fraction'] = window_data['anomaly'].mean()
        
        # Multiclass classification (event type)
        if 'event' in window_data.columns:
            event_counts = window_data['event'].value_counts()
            labels['multiclass_label'] = event_counts.index[0]  # Majority class
            labels['label_confidence'] = event_counts.iloc[0] / len(window_data)
            labels['unique_events'] = len(event_counts)
        
        return labels
    
    def extract_window_features(self, window_data: pd.DataFrame) -> Dict:
        """
        Extract features from window data.
        
        Args:
            window_data: DataFrame containing window data
            
        Returns:
            Dictionary with extracted features
        """
        try:
            features = {}
            
            # Extract vital signs
            vital_features = []
            for col in self.vital_columns:
                if col in window_data.columns:
                    values = window_data[col].dropna().values
                    if len(values) > 0:
                        vital_features.append(values)
                    else:
                        # Use zeros for missing vital signs
                        vital_features.append(np.zeros(self.window_size))
            
            if vital_features:
                # Pad or truncate to window_size
                padded_vitals = []
                for vital in vital_features:
                    if len(vital) < self.window_size:
                        # Pad with last value or zero
                        pad_value = vital[-1] if len(vital) > 0 else 0
                        padded = np.pad(vital, (0, self.window_size - len(vital)), 
                                      mode='constant', constant_values=pad_value)
                    else:
                        padded = vital[:self.window_size]
                    padded_vitals.append(padded)
                
                features['vital_signs'] = np.array(padded_vitals).T  # Shape: (window_size, n_vitals)
            
            # Extract waveforms
            waveform_features = []
            for col in self.waveform_columns:
                if col in window_data.columns:
                    waveforms = []
                    expected_length = 80  # Default from prompt
                    
                    for waveform in window_data[col]:
                        if waveform is not None and not np.all(np.isnan(waveform)):
                            # Apply preprocessing
                            signal_type = 'ecg' if 'ecg' in col.lower() else 'ppg'
                            processed = self.preprocess_waveforms(waveform, signal_type)
                            
                            # Ensure consistent length
                            if len(processed) != expected_length:
                                if len(processed) > expected_length:
                                    processed = processed[:expected_length]
                                else:
                                    processed = np.pad(processed, (0, expected_length - len(processed)), 
                                                     mode='constant', constant_values=0)
                            waveforms.append(processed)
                        else:
                            # Create zero waveform for missing data
                            waveforms.append(np.zeros(expected_length))
                    
                    if waveforms:
                        waveform_features.append(np.array(waveforms))
            
            if waveform_features:
                # Stack waveforms: Shape (window_size, waveform_length, n_channels)
                try:
                    features['waveforms'] = np.stack(waveform_features, axis=-1)
                except ValueError as e:
                    # If stacking fails, store each waveform type separately
                    for i, col in enumerate(self.waveform_columns):
                        if i < len(waveform_features):
                            features[f'waveform_{col.lower()}'] = waveform_features[i]
            
            # Extract PSG signals - handle different sampling rates separately
            psg_features = {}
            for col in self.psg_columns:
                if col in window_data.columns:
                    psg_signals = []
                    expected_length = {'PSG_Flow': 20, 'PSG_Snore': 10, 'PSG_Position': 10,
                                     'PSG_Thorax': 10, 'PSG_Abdomen': 10}.get(col, 10)
                    
                    for psg_signal in window_data[col]:
                        if psg_signal is not None and not np.all(np.isnan(psg_signal)):
                            processed = self.preprocess_waveforms(psg_signal, 'psg')
                            # Ensure consistent length
                            if len(processed) != expected_length:
                                if len(processed) > expected_length:
                                    processed = processed[:expected_length]
                                else:
                                    processed = np.pad(processed, (0, expected_length - len(processed)), 
                                                     mode='constant', constant_values=0)
                            psg_signals.append(processed)
                        else:
                            # Create zero signal for missing data
                            psg_signals.append(np.zeros(expected_length))
                    
                    if psg_signals:
                        # Store each PSG signal type separately
                        features[f'psg_{col.lower()}'] = np.array(psg_signals)
            
            # Also create a combined PSG representation if needed
            if any(key.startswith('psg_') for key in features.keys()):
                features['has_psg_signals'] = True
            
            return features
            
        except Exception as e:
            warnings.warn(f"Feature extraction failed: {e}")
            # Return minimal features to continue processing
            return {
                'vital_signs': np.zeros((self.window_size, len(self.vital_columns))),
                'error': str(e)
            }
    
    def normalize_patient_data(self, patient_windows: List[Dict], patient_id: int) -> List[Dict]:
        """
        Apply per-patient normalization to vital signs.
        
        Args:
            patient_windows: List of windows for a patient
            patient_id: Patient identifier
            
        Returns:
            List of windows with normalized features
        """
        if not patient_windows:
            return patient_windows
        
        # Collect all vital signs for this patient
        all_vitals = []
        for window in patient_windows:
            if 'vital_signs' in window:
                all_vitals.append(window['vital_signs'])
        
        if not all_vitals:
            return patient_windows
        
        # Stack all vital signs
        stacked_vitals = np.concatenate(all_vitals, axis=0)
        
        # Fit scaler on patient data
        scaler = StandardScaler()
        scaler.fit(stacked_vitals)
        self.patient_scalers[patient_id] = scaler
        
        # Apply normalization to each window
        normalized_windows = []
        for window in patient_windows:
            window_copy = window.copy()
            if 'vital_signs' in window_copy:
                normalized_vitals = scaler.transform(window_copy['vital_signs'])
                window_copy['vital_signs'] = normalized_vitals
            normalized_windows.append(window_copy)
        
        return normalized_windows
    
    def process_dataset(self, dataset: pd.DataFrame, output_path: Optional[str] = None) -> List[Dict]:
        """
        Process the entire dataset into windows.
        
        Args:
            dataset: Input dataset DataFrame
            output_path: Optional path to save processed data
            
        Returns:
            List of all processed windows
        """
        all_windows = []
        patients = dataset['patient'].unique()
        
        print(f"Processing {len(patients)} patients...")
        
        for patient_id in tqdm(patients, desc="Processing patients"):
            patient_data = dataset[dataset['patient'] == patient_id].copy()
            
            # Create windows for this patient
            patient_windows = self.create_windows(patient_data)
            
            if patient_windows:
                # Apply per-patient normalization
                normalized_windows = self.normalize_patient_data(patient_windows, patient_id)
                all_windows.extend(normalized_windows)
        
        print(f"Generated {len(all_windows)} windows from {len(patients)} patients")
        
        # Save processed data if path provided
        if output_path:
            self.save_processed_data(all_windows, output_path)
        
        return all_windows
    
    def save_processed_data(self, windows: List[Dict], output_path: str):
        """Save processed windows to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create metadata
        metadata = {
            'n_windows': len(windows),
            'window_size': self.window_size,
            'overlap': self.overlap,
            'filter_params': {
                'order': self.filter_order,
                'low_freq': self.filter_low,
                'high_freq': self.filter_high
            },
            'missing_threshold': self.missing_threshold,
            'patient_scalers': self.patient_scalers
        }
        
        data_to_save = {
            'windows': windows,
            'metadata': metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Saved processed data to {output_path}")
    
    @classmethod
    def load_processed_data(cls, file_path: str) -> Tuple[List[Dict], Dict]:
        """Load processed data from file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['windows'], data['metadata']


def main():
    """Example usage of the preprocessor."""
    # Example configuration
    preprocessor = OSASPreprocessor(
        window_size=60,
        overlap=0.5,
        filter_low=5.0,
        filter_high=35.0,
        missing_threshold=0.5
    )
    
    # Load dataset
    try:
        dataset = pd.read_pickle('./data/dataset_OSAS.pickle')
        print(f"Loaded dataset with {len(dataset)} records")
        
        # Process dataset
        processed_windows = preprocessor.process_dataset(
            dataset, 
            output_path='./data/processed_windows.pkl'
        )
        
        print(f"Preprocessing complete. Generated {len(processed_windows)} windows.")
        
        # Print sample window info
        if processed_windows:
            sample_window = processed_windows[0]
            print(f"\nSample window keys: {list(sample_window.keys())}")
            if 'vital_signs' in sample_window:
                print(f"Vital signs shape: {sample_window['vital_signs'].shape}")
            if 'waveforms' in sample_window:
                print(f"Waveforms shape: {sample_window['waveforms'].shape}")
        
    except FileNotFoundError:
        print("Dataset file not found. Please ensure dataset_OSAS.pickle is available.")
        print("Run this script after placing the dataset in ./data/dataset_OSAS.pickle")


if __name__ == "__main__":
    main()
