"""
TFN Configuration for MIMIC-IV
==============================

Configuration parameters adapted from the original TFN paper
for MIMIC-IV electronic health records instead of NephroCAGE.

Key adaptations:
- Static features: MIMIC-IV demographics (gender, race, admission_type)
- Time series: MIMIC-IV vitals (heart_rate, bp, temperature, etc.)
- Notes: Smaller GTE-base model for efficiency
"""

# TFN (Temporal Fusion Nexus) Configuration
# Adapted for MIMIC-IV from original NephroCAGE config

TFN_CONFIG = {
    # MIMIC-IV adapted features
    'static_categorical_cols': ['gender', 'race', 'admission_type'],
    'static_numerical_cols': ['age', 'los'],  # Length of stay, etc.

    # MIMIC-IV vital signs and labs
    'ts_features': ['heart_rate', 'sbp', 'dbp', 'temperature', 'resp_rate', 'spo2'],

    # Model dimensions
    'static_embedding_dim': 16,
    'static_output_dim': 256,
    'lstm_hidden_size': 512,
    'lstm_num_layers': 2,
    'num_heads': 8,
    'PADDING_VAL': 0,
    'notes_embedding_dim': 768,  # Smaller for efficiency

    # Additional model parameters
    'input_dim': len(['heart_rate', 'sbp', 'dbp', 'temperature', 'resp_rate', 'spo2']),  # 6 vitals
    'output_dim': 256,
    'dropout': 0.1,
}