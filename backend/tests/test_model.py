from backend.model import classify_voice
import pytest

def test_classify_voice_human_like():
    # Simulate features with high variance (expressive human)
    features = {
        "pitch_var": 800.0,
        "mfcc_var": [60.0] * 13,
        "spectral_flatness_mean": 0.01,
        "rms_var": 0.05
    }
    
    result = classify_voice(features)
    assert result["classification"] == "HUMAN"
    assert result["confidence"] > 0.5
    assert "details" in result

def test_classify_voice_ai_like():
    # Simulate features with low variance (monotonic AI)
    features = {
        "pitch_var": 50.0,
        "mfcc_var": [10.0] * 13,
        "spectral_flatness_mean": 0.001,
        "rms_var": 0.01
    }
    
    result = classify_voice(features)
    assert result["classification"] == "AI_GENERATED"
    assert result["confidence"] > 0.5

def test_classify_voice_boundary():
    # Edge case: zero variance
    features = {
        "pitch_var": 0.0,
        "mfcc_var": [0.0] * 13
    }
    
    result = classify_voice(features)
    assert result["classification"] == "AI_GENERATED"
    assert result["confidence"] > 0.9 # Should be very confident it's not human
