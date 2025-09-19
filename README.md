# OSAS Detection with Transformer-based Deep Learning

## Project Overview

This project implements a deep learning pipeline for detecting **Obstructive Sleep Apnea Syndrome (OSAS)** from time-series vital signs and waveform data using transformer architecture. The system processes multimodal physiological signals to classify breathing anomalies and specific apnea event types.

### **Objectives**

- **Binary Classification**: Detect breathing anomalies (Normal vs. Anomaly)
- **Multi-class Classification**: Classify specific event types:
  - NONE
  - HYPOPNEA
  - APNEA-OBSTRUCTIVE
  - APNEA-MIXED
  - APNEA-CENTRAL

## **Dataset: OSASUD**

- **Source**: 30 stroke patients from Udine University Hospital
- **Format**: Pandas DataFrame with 18 columns
- **Resolution**: 1-second time granularity
- **Labels**: Physician-provided per AASM scoring rules
- **Total Records**: ~961,357 samples
- **Windows Generated**: ~31,986 (60-second windows with 50% overlap)

### **Data Features**

| Category        | Features                               | Sampling Rate |
| --------------- | -------------------------------------- | ------------- |
| **Vital Signs** | HR, SpO2, PI, RR, PVCs                 | 1 Hz          |
| **Waveforms**   | ECG (3 leads), PPG                     | 80 Hz         |
| **PSG Signals** | Flow, Snore, Position, Thorax, Abdomen | 10-20 Hz      |
| **Labels**      | Binary (anomaly), Multi-class (event)  | 1 Hz          |

## **Architecture**

### **Transformer Model Components**

1. **MultiModalEncoder**: Processes different signal types
   - Vital signs: Linear projection to model dimension
   - Waveforms: 1D CNN + adaptive pooling + projection
   - PSG signals: Adaptive encoders for variable lengths
2. **PositionalEncoding**: Sinusoidal encoding for temporal patterns
3. **OSASTransformer**: Multi-head attention with encoder layers
4. **Classification Heads**:
   - Binary: Normal vs. Anomaly detection
   - Multi-class: 5-class event type classification
   - Multi-task: Combined binary + multi-class learning

### **Key Features**

- **Multi-modal Processing**: Handles vital signs, waveforms, and PSG signals
- **Attention Mechanism**: Self-attention learns temporal dependencies
- **Class Imbalance Handling**: Weighted loss, early stopping on macro-F1
- **Patient-aware Splitting**: Prevents data leakage across patients
- **Flexible Architecture**: Supports binary, multiclass, and multitask learning
- **Model Checkpointing**: Separate directories for models and results

## **Project Structure**

```
osas_detection_transformer/
├── data/                           # Data directory (not tracked)
│   └── dataset_OSAS.pickle         # Raw dataset file
├── checkpoints/                    # Model outputs
│   ├── models/                     # Trained models (not tracked)
│   │   ├── best_model_binary.pth
│   │   ├── best_model_multiclass.pth
│   │   └── best_model_multitask.pth
│   └── results/                    # Training results (tracked)
│       ├── training_results_binary.json
│       ├── training_results_multiclass.json
│       └── training_results_multitask.json
├── eda_basic_statistics.ipynb     # Basic dataset statistics
├── eda_class_distribution.ipynb   # Class balance analysis
├── eda_signal_quality.ipynb       # Signal quality assessment
├── eda_label_alignment.ipynb      # Label consistency analysis
├── preprocess_dataset.ipynb       # Data preprocessing notebook
├── evaluate_model.ipynb           # Model evaluation notebook
├── dataset.py                     # Dataset classes and data loading
├── model.py                       # Transformer model architecture
├── preprocess_data.py             # Data preprocessing utilities
├── train_model.py                 # Training script with OSASTrainer
├── requirements.txt               # Python dependencies
├── dataset_description.pdf        # Dataset documentation
└── README.md                      # This file
```

## **Getting Started**

### **1. Installation**

```bash
pip install -r requirements.txt
```

### **2. Data Preparation**

```bash
# Place dataset_OSAS.pickle in ./data/ directory
# Run preprocessing
python preprocess_data.py --data_path ./data/dataset_OSAS.pickle --output_path ./data/processed_windows.pkl
```

### **3. Exploratory Data Analysis**

```bash
# Run EDA notebooks in order:
jupyter notebook eda_basic_statistics.ipynb
jupyter notebook eda_class_distribution.ipynb
jupyter notebook eda_signal_quality.ipynb
jupyter notebook eda_label_alignment.ipynb
```

### **4. Model Training**

```bash
# Binary classification
python train_model.py --task binary --epochs 100 --batch_size 32 --learning_rate 1e-4

# Multi-class classification
python train_model.py --task multiclass --epochs 100 --batch_size 32 --learning_rate 1e-4

# Multi-task learning
python train_model.py --task multitask --epochs 100 --batch_size 32 --learning_rate 1e-4

# With custom configuration
python train_model.py --task binary --d_model 256 --nhead 8 --num_encoder_layers 6 --use_wandb --experiment_name "osas_binary_v1"
```

### **5. Model Evaluation**

```bash
jupyter notebook evaluate_model.ipynb
```

## **Configuration**

### **Model Parameters**

```python
model_config = {
    'vital_dim': 5,                  # Vital signs features (HR, SpO2, PI, RR, PVCs)
    'waveform_channels': 4,          # ECG (3 leads) + PPG waveforms
    'waveform_length': 80,           # 80 Hz sampling rate
    'psg_channels': 5,               # PSG signal channels
    'psg_lengths': [20, 10, 10, 10, 10],  # Variable PSG lengths
    'd_model': 256,                  # Model dimension
    'nhead': 8,                      # Number of attention heads
    'num_encoder_layers': 6,         # Transformer encoder layers
    'dropout': 0.1,                  # Dropout rate
    'task': 'binary',                # 'binary', 'multiclass', or 'multitask'
    'num_classes': 2,                # 2 for binary, 5 for multiclass
    'max_sequence_length': 60,       # Maximum sequence length
    'use_cls_token': True            # Use classification token
}
```

### **Training Parameters**

```python
training_config = {
    'learning_rate': 1e-4,           # Learning rate
    'weight_decay': 0.01,            # AdamW weight decay
    'epochs': 100,                   # Maximum epochs
    'batch_size': 32,                # Batch size
    'early_stopping_patience': 15,   # Early stopping patience
    'scheduler': 'cosine',           # LR scheduler type
    'use_class_weights': True,       # Handle class imbalance
    'grad_clip': 1.0,                # Gradient clipping
    'save_dir': './checkpoints'      # Checkpoint directory
}
```

## **Preprocessing Pipeline**

1. **Data Loading**: Load OSASUD dataset from pickle file
2. **Signal Filtering**: 2nd-order Butterworth bandpass (5-35 Hz) for ECG/PPG
3. **Windowing**: 60-second windows with optional overlap
4. **Quality Control**: Drop windows with >50% missing data
5. **Normalization**: Per-patient Z-score normalization for vital signs
6. **Feature Engineering**: Extract waveform features and PSG signals
7. **Label Assignment**: Binary/multi-class labels per window
8. **Patient-aware Splitting**: Ensure no patient data leakage between splits

## **Model Output Structure**

After training, the following structure is created:

```
checkpoints/
├── models/                         # Trained model files (not tracked in git)
│   ├── best_model_binary.pth      # Best binary classification model
│   ├── best_model_multiclass.pth  # Best multiclass classification model
│   └── best_model_multitask.pth   # Best multitask learning model
└── results/                        # Training results (tracked in git)
    ├── training_results_binary.json
    ├── training_results_multiclass.json
    └── training_results_multitask.json
```

## **Class Imbalance Handling**

⚠️ **Highly Imbalanced Dataset**:

- Normal: ~87%
- Anomaly: ~13%
- Rare classes (APNEA-CENTRAL): <1%

**Strategies Applied**:

- Stratified sampling
- Class-weighted loss functions
- Focal loss for rare classes
- SMOTE oversampling
- Macro-F1 evaluation metric

## **Evaluation Metrics**

- **Primary**: Macro F1-Score (handles imbalance)
- **Secondary**: Per-class Precision/Recall
- **Clinical**: Sensitivity for apnea detection
- **Visualization**: Confusion matrix, ROC curves

## **Research Extensions**

### **Implemented Features**

- ✅ Multi-modal fusion (vital signs + waveforms + PSG)
- ✅ Multi-task learning (binary + multiclass simultaneously)
- ✅ Attention mechanism with positional encoding
- ✅ Patient-aware data splitting
- ✅ Class imbalance handling with weighted losses
- ✅ Early stopping with macro-F1 metric
- ✅ Learning rate scheduling (cosine, step, plateau)
- ✅ Comprehensive evaluation metrics
- ✅ Model checkpointing with task-specific naming
- ✅ Attention weight extraction and visualization
- ✅ Per-patient performance analysis
- ✅ Prediction confidence analysis

### **Current Capabilities**

- **OSASTrainer Class**: Complete training pipeline with metrics tracking
- **MultiModalEncoder**: Handles different input modalities efficiently
- **OSASTransformer**: Transformer architecture with flexible heads
- **Patient-aware Splitting**: Prevents data leakage
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, calibration plots

### **Future Enhancements**

- 🔄 Self-supervised pre-training on unlabeled physiological data
- 🔄 SHAP/LIME interpretability analysis
- 🔄 Real-time inference optimization
- 🔄 Sequence labeling for event boundary detection
- 🔄 Cross-patient domain adaptation
- 🔄 Federated learning for multi-hospital deployment
- 🔄 Integration with clinical decision support systems

## **Troubleshooting**

### **Common Issues**

1. **Out of Memory**:

   - Reduce `batch_size` (try 16 or 8)
   - Reduce `d_model` or `num_encoder_layers`
   - Use gradient checkpointing

2. **Poor Performance**:

   - Check class weights: `dataset.get_class_weights()`
   - Verify data preprocessing in EDA notebooks
   - Increase `early_stopping_patience`
   - Try different learning rates (1e-5 to 1e-3)

3. **NaN Loss**:

   - Lower learning rate (1e-5)
   - Check input normalization
   - Enable gradient clipping: `--grad_clip 1.0`
   - Verify no infinite values in data

4. **Overfitting**:
   - Increase dropout rate (`--dropout 0.2`)
   - Reduce model complexity
   - Use stronger weight decay (`--weight_decay 0.1`)

### **Data Issues**

- **Missing Dataset**: Ensure `dataset_OSAS.pickle` is in `./data/` directory
- **Preprocessing Errors**: Run EDA notebooks to check signal quality
- **Patient ID Issues**: Verify patient consistency in `dataset.py`
- **Memory Issues**: Process smaller batches or reduce window overlap

### **Model Issues**

- **Training Stuck**: Check learning rate and scheduler settings
- **Poor Validation**: Verify patient-aware splitting is working
- **Attention Errors**: Ensure sequence lengths are compatible
- **Multi-task Convergence**: Balance loss weights for binary/multiclass tasks

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research implementation. Clinical validation and regulatory approval are required for medical use.
