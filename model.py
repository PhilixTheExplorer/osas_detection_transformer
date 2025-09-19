"""
Transformer-based model for OSAS detection.

This module implements a transformer architecture for processing multimodal
time-series data (vital signs, ECG/PPG waveforms, PSG signals) for sleep
apnea detection and classification.

Supports:
- Binary classification (anomaly detection)
- Multiclass classification (event type prediction)
- Multimodal input processing
- Positional encoding for temporal patterns
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiModalEncoder(nn.Module):
    """Encoder for different input modalities."""
    
    def __init__(self, 
                 vital_dim: int = 5,
                 waveform_channels: int = 4,
                 waveform_length: int = 80,
                 psg_channels: int = 5,
                 psg_lengths: List[int] = [20, 10, 10, 10, 10],
                 d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Vital signs encoder (simple linear projection)
        self.vital_encoder = nn.Linear(vital_dim, d_model) if vital_dim > 0 else None
        
        # Waveform encoder (1D CNN + projection)
        if waveform_channels > 0:
            self.waveform_conv = nn.Sequential(
                nn.Conv1d(waveform_channels, 64, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, d_model)
            )
        else:
            self.waveform_conv = None
        
        # PSG signals encoder (adaptive for different lengths)
        if psg_channels > 0:
            self.psg_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=min(5, length), padding=min(2, length//2)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, d_model // psg_channels)
                ) for length in psg_lengths
            ])
            self.psg_projection = nn.Linear(d_model, d_model)
        else:
            self.psg_encoders = None
            self.psg_projection = None
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multimodal features.
        
        Args:
            features: Dictionary containing different modality features
            
        Returns:
            Encoded tensor of shape (batch_size, sequence_length, d_model)
        """
        encoded_features = []
        batch_size = None
        sequence_length = None
        
        # Process vital signs
        if 'vital_signs' in features and self.vital_encoder is not None:
            vital_signs = features['vital_signs']  # (batch, seq_len, vital_dim)
            batch_size, sequence_length, _ = vital_signs.shape
            
            # Encode each time step
            vital_encoded = self.vital_encoder(vital_signs)  # (batch, seq_len, d_model)
            encoded_features.append(vital_encoded)
        
        # Process waveforms
        if 'waveforms' in features and self.waveform_conv is not None:
            waveforms = features['waveforms']  # (batch, seq_len, waveform_len, channels)
            batch_size, sequence_length, waveform_len, channels = waveforms.shape
            
            # Reshape for processing each time step
            waveforms_flat = waveforms.view(-1, waveform_len, channels)  # (batch*seq_len, waveform_len, channels)
            waveforms_flat = waveforms_flat.transpose(1, 2)  # (batch*seq_len, channels, waveform_len)
            
            # Apply convolution
            waveform_encoded = self.waveform_conv(waveforms_flat)  # (batch*seq_len, d_model)
            waveform_encoded = waveform_encoded.view(batch_size, sequence_length, self.d_model)
            
            encoded_features.append(waveform_encoded)
        
        # Process PSG signals
        if 'psg_signals' in features and self.psg_encoders is not None:
            psg_signals = features['psg_signals']  # (batch, seq_len, max_psg_len, psg_channels)
            batch_size, sequence_length, max_psg_len, psg_channels = psg_signals.shape
            
            psg_encoded_parts = []
            for i, encoder in enumerate(self.psg_encoders):
                if i < psg_channels:
                    # Extract channel i
                    psg_channel = psg_signals[:, :, :, i]  # (batch, seq_len, max_psg_len)
                    psg_channel_flat = psg_channel.view(-1, 1, max_psg_len)  # (batch*seq_len, 1, max_psg_len)
                    
                    # Encode
                    encoded_part = encoder(psg_channel_flat)  # (batch*seq_len, d_model//psg_channels)
                    encoded_part = encoded_part.view(batch_size, sequence_length, -1)
                    psg_encoded_parts.append(encoded_part)
            
            if psg_encoded_parts:
                psg_encoded = torch.cat(psg_encoded_parts, dim=-1)  # (batch, seq_len, d_model)
                psg_encoded = self.psg_projection(psg_encoded)
                encoded_features.append(psg_encoded)
        
        # Handle case where we only have one modality or need to determine dimensions
        if not encoded_features:
            # Return zero tensor if no features
            if batch_size is None:
                batch_size = 1
                sequence_length = 60  # Default window size
            return torch.zeros(batch_size, sequence_length, self.d_model)
        
        # Sum or concatenate encoded features
        if len(encoded_features) == 1:
            return encoded_features[0]
        else:
            # Average multiple modalities
            stacked = torch.stack(encoded_features, dim=0)
            return torch.mean(stacked, dim=0)


class OSASTransformer(nn.Module):
    """Transformer model for OSAS detection."""
    
    def __init__(self,
                 # Input dimensions
                 vital_dim: int = 5,
                 waveform_channels: int = 4,
                 waveform_length: int = 80,
                 psg_channels: int = 5,
                 psg_lengths: List[int] = [20, 10, 10, 10, 10],
                 
                 # Model dimensions
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 
                 # Task configuration
                 task: str = 'binary',
                 num_classes: int = 2,
                 
                 # Additional parameters
                 max_sequence_length: int = 60,
                 use_cls_token: bool = True):
        """
        Initialize OSAS Transformer model.
        
        Args:
            vital_dim: Number of vital sign features
            waveform_channels: Number of waveform channels (ECG + PPG)
            waveform_length: Length of each waveform sample
            psg_channels: Number of PSG signal channels
            psg_lengths: Lengths of different PSG signals
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            task: 'binary' or 'multiclass'
            num_classes: Number of output classes
            max_sequence_length: Maximum sequence length
            use_cls_token: Whether to use CLS token for classification
        """
        super().__init__()
        
        self.task = task
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.max_sequence_length = max_sequence_length
        
        # Multimodal encoder
        self.encoder = MultiModalEncoder(
            vital_dim=vital_dim,
            waveform_channels=waveform_channels,
            waveform_length=waveform_length,
            psg_channels=psg_channels,
            psg_lengths=psg_lengths,
            d_model=d_model
        )
        
        # CLS token for classification
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_sequence_length + 1, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor], 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            features: Dictionary of input features
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing logits and attention weights
        """
        # Encode features
        encoded = self.encoder(features)  # (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = encoded.shape
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
            encoded = torch.cat([cls_tokens, encoded], dim=1)  # (batch_size, seq_len+1, d_model)
            
            # Update attention mask
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Apply positional encoding
        encoded = encoded.transpose(0, 1)  # (seq_len, batch_size, d_model) for pos encoding
        encoded = self.pos_encoder(encoded)
        encoded = encoded.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert to attention mask format (True = attend, False = ignore)
            attention_mask = attention_mask.bool()
            # Invert for transformer (True = ignore, False = attend)
            attention_mask = ~attention_mask
        
        transformer_output = self.transformer(encoded, src_key_padding_mask=attention_mask)
        
        # Extract features for classification
        if self.use_cls_token:
            # Use CLS token output
            cls_output = transformer_output[:, 0, :]  # (batch_size, d_model)
        else:
            # Use average pooling over sequence
            if attention_mask is not None:
                # Masked average pooling
                mask = (~attention_mask).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
                masked_output = transformer_output * mask
                cls_output = masked_output.sum(dim=1) / mask.sum(dim=1)
            else:
                cls_output = transformer_output.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        
        # Return outputs
        outputs = {
            'logits': logits,
            'hidden_states': transformer_output,
            'attention_weights': None  # Could extract from transformer if needed
        }
        
        return outputs
    
    def get_attention_weights(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract attention weights for interpretability."""
        # This would require modifying the transformer to return attention weights
        # For now, return None
        return None


class OSASMultiTaskTransformer(OSASTransformer):
    """Multi-task version for joint binary and multiclass prediction."""
    
    def __init__(self, *args, **kwargs):
        # Override num_classes and task
        kwargs['task'] = 'multitask'
        kwargs['num_classes'] = 2  # Will be overridden
        
        super().__init__(*args, **kwargs)
        
        # Replace classifier with separate heads
        self.binary_classifier = nn.Sequential(
            nn.Dropout(self.classifier[0].p),
            nn.Linear(self.d_model, self.classifier[1].in_features),
            nn.ReLU(),
            nn.Dropout(self.classifier[3].p),
            nn.Linear(self.classifier[1].out_features, 2)  # Binary classification
        )
        
        self.multiclass_classifier = nn.Sequential(
            nn.Dropout(self.classifier[0].p),
            nn.Linear(self.d_model, self.classifier[1].in_features),
            nn.ReLU(),
            nn.Dropout(self.classifier[3].p),
            nn.Linear(self.classifier[1].out_features, 5)  # 5-class classification
        )
        
        # Remove original classifier
        del self.classifier
    
    def forward(self, features: Dict[str, torch.Tensor], 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task model."""
        # Get hidden representations
        encoded = self.encoder(features)
        batch_size, seq_len, _ = encoded.shape
        
        # Add CLS token and apply transformer (same as parent)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            encoded = torch.cat([cls_tokens, encoded], dim=1)
            
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        encoded = encoded.transpose(0, 1)
        encoded = self.pos_encoder(encoded)
        encoded = encoded.transpose(0, 1)
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        transformer_output = self.transformer(encoded, src_key_padding_mask=attention_mask)
        
        # Extract CLS representation
        if self.use_cls_token:
            cls_output = transformer_output[:, 0, :]
        else:
            if attention_mask is not None:
                mask = (~attention_mask).float().unsqueeze(-1)
                masked_output = transformer_output * mask
                cls_output = masked_output.sum(dim=1) / mask.sum(dim=1)
            else:
                cls_output = transformer_output.mean(dim=1)
        
        # Dual predictions
        binary_logits = self.binary_classifier(cls_output)
        multiclass_logits = self.multiclass_classifier(cls_output)
        
        return {
            'binary_logits': binary_logits,
            'multiclass_logits': multiclass_logits,
            'hidden_states': transformer_output,
            'attention_weights': None
        }


def create_model(config: Dict) -> OSASTransformer:
    """
    Factory function to create OSAS transformer model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured model instance
    """
    # Extract only model-related parameters
    model_params = {
        'vital_dim': config.get('vital_dim', 5),
        'waveform_channels': config.get('waveform_channels', 4),
        'waveform_length': config.get('waveform_length', 80),
        'psg_channels': config.get('psg_channels', 5),
        'psg_lengths': config.get('psg_lengths', [20, 10, 10, 10, 10]),
        'd_model': config.get('d_model', 256),
        'nhead': config.get('nhead', 8),
        'num_encoder_layers': config.get('num_encoder_layers', 6),
        'dim_feedforward': config.get('dim_feedforward', 1024),
        'dropout': config.get('dropout', 0.1),
        'task': config.get('task', 'binary'),
        'num_classes': config.get('num_classes', 2),
        'max_sequence_length': config.get('max_sequence_length', 60),
        'use_cls_token': config.get('use_cls_token', True)
    }
    
    task = config.get('task', 'binary')
    
    if task == 'multitask':
        return OSASMultiTaskTransformer(**model_params)
    else:
        return OSASTransformer(**model_params)


def main():
    """Example usage and testing of the model."""
    # Example configuration
    config = {
        'vital_dim': 5,
        'waveform_channels': 4,
        'waveform_length': 80,
        'psg_channels': 5,
        'psg_lengths': [20, 10, 10, 10, 10],
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'task': 'binary',
        'num_classes': 2,
        'max_sequence_length': 60,
        'use_cls_token': True
    }
    
    # Create model
    model = create_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy data
    batch_size = 4
    seq_len = 60
    
    dummy_features = {
        'vital_signs': torch.randn(batch_size, seq_len, 5),
        'waveforms': torch.randn(batch_size, seq_len, 80, 4),
        'psg_signals': torch.randn(batch_size, seq_len, 20, 5)
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_features)
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Test multi-task model
    config['task'] = 'multitask'
    multitask_model = create_model(config)
    
    with torch.no_grad():
        outputs = multitask_model(dummy_features)
        print(f"Binary logits shape: {outputs['binary_logits'].shape}")
        print(f"Multiclass logits shape: {outputs['multiclass_logits'].shape}")
    
    print("Model testing completed successfully!")


if __name__ == "__main__":
    main()
