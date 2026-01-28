import numpy as np
import pytest
from backend.features import extract_features

def test_extract_features_shape():
    # Generate 1 second of random noise at 22050Hz
    sr = 22050
    y = np.random.uniform(-1, 1, sr)
    
    features = extract_features(y, sr)
    
    # Check if all keys exist
    assert "mfcc_mean" in features
    assert "mfcc_var" in features
    assert "pitch_var" in features
    assert "spectral_flatness_mean" in features
    assert "rms_var" in features
    
    # Check types and shapes
    assert len(features["mfcc_mean"]) == 13
    assert len(features["mfcc_var"]) == 13
    assert isinstance(features["pitch_var"], float)
    assert isinstance(features["spectral_flatness_mean"], float)
    assert isinstance(features["rms_var"], float)

def test_extract_features_silent():
    # Generate 1 second of silence
    sr = 22050
    y = np.zeros(sr)
    
    features = extract_features(y, sr)
    
    # Pitch variance should be 0.0 for silence (no voiced parts)
    assert features["pitch_var"] == 0.0
    # RMS variance should be 0.0 for silence
    assert features["rms_var"] == 0.0
