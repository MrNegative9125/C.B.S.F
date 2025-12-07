"""
USAGE EXAMPLE - Cyberattack Prediction with Attack Type Detection
Trained on COMPLETE CICIDS 2018 Dataset with K-Fold Cross-Validation
"""

import pickle
import pandas as pd
import numpy as np
import json

# Load models
with open('dwig_ML9125/models/binary_model_Random_Forest.pkl', 'rb') as f:
    binary_model = pickle.load(f)

with open('dwig_ML9125/models/multiclass_model.pkl', 'rb') as f:
    multiclass_model = pickle.load(f)

with open('dwig_ML9125/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('dwig_ML9125/models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('dwig_ML9125/models/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

with open('dwig_ML9125/models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✓ All models loaded successfully!")
print(f"Model trained on COMPLETE CICIDS 2018 dataset")
print(f"Total attack types: {len(label_mapping)}")

# Example prediction on new data
# Step 1: Load your new network traffic data
# new_data = pd.read_csv('new_network_data.csv')

# Step 2: Ensure it has the same features
# new_data = new_data[feature_names]

# Step 3: Scale the data
# new_data_scaled = scaler.transform(new_data)

# Step 4: Binary prediction (Is it an attack?)
# attack_proba = binary_model.predict_proba(new_data_scaled)[:, 1]
# is_attack = binary_model.predict(new_data_scaled)

# Step 5: Multi-class prediction (What type of attack?)
# attack_type_encoded = multiclass_model.predict(new_data_scaled)
# attack_type_names = [label_mapping[str(i)] for i in attack_type_encoded]
# attack_confidence = multiclass_model.predict_proba(new_data_scaled).max(axis=1)

# Step 6: Create results dataframe
# results = pd.DataFrame({
#     'attack_probability': attack_proba,
#     'is_attack': is_attack,
#     'predicted_attack_type': attack_type_names,
#     'prediction_confidence': attack_confidence,
#     'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in attack_proba]
# })

# print(results.head())

# Example with single sample:
def predict_single_sample(sample_features):
    """
    Predict for a single network traffic sample

    Args:
        sample_features: dict or array of features

    Returns:
        dict with prediction results
    """
    # Convert to dataframe if dict
    if isinstance(sample_features, dict):
        sample_df = pd.DataFrame([sample_features])
    else:
        sample_df = pd.DataFrame([sample_features], columns=feature_names)

    # Scale
    sample_scaled = scaler.transform(sample_df)

    # Binary prediction
    attack_prob = binary_model.predict_proba(sample_scaled)[0, 1]
    is_attack = binary_model.predict(sample_scaled)[0]

    # Multi-class prediction
    attack_type_idx = multiclass_model.predict(sample_scaled)[0]
    attack_type = label_mapping[str(attack_type_idx)]
    confidence = multiclass_model.predict_proba(sample_scaled)[0].max()

    # Risk assessment
    if attack_prob > 0.7:
        risk = "HIGH"
    elif attack_prob > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        'is_attack': bool(is_attack),
        'attack_probability': float(attack_prob),
        'attack_type': attack_type,
        'confidence': float(confidence),
        'risk_level': risk
    }

# Example usage:
# result = predict_single_sample(sample_features)
# print(f"Attack Detected: {result['is_attack']}")
# print(f"Attack Probability: {result['attack_probability']:.2%}")
# print(f"Attack Type: {result['attack_type']}")
# print(f"Confidence: {result['confidence']:.2%}")
# print(f"Risk Level: {result['risk_level']}")
