import numpy as np

def classify_voice(features: dict) -> dict:
    """
    Classifies voice as 'AI_GENERATED' or 'HUMAN' based on acoustic features.
    
    Heuristic Logic:
    1. Pitch Variance (pitch_var): Human speech typically has higher pitch variability 
       (intonation, micro-tremors) compared to standard TTS systems which may be more monotonic.
    2. MFCC Variance (mfcc_var): Represents the variability of the spectral envelope. 
       Natural speech tends to have higher spectral variability due to coarticulation and natural dynamics.
       
    This is a simplified heuristic and not a trained ML model.
    """
    
    # Extract relevant features
    pitch_var = features.get("pitch_var", 0.0)
    mfcc_var = features.get("mfcc_var", [])
    
    # Calculate average MFCC variance if available
    avg_mfcc_var = np.mean(mfcc_var) if mfcc_var else 0.0
    
    # Heuristic thresholds (estimated values for this prototype)
    PITCH_VAR_THRESHOLD = 500.0
    MFCC_VAR_THRESHOLD = 50.0
    
    # Calculate "Human-ness" scores (0.0 to 1.0)
    # We use tanh to normalize inputs to 0-1 range softly
    pitch_score = np.tanh(pitch_var / PITCH_VAR_THRESHOLD)
    mfcc_score = np.tanh(avg_mfcc_var / MFCC_VAR_THRESHOLD)
    
    # Weighted combination
    # Giving more weight to pitch variance as it's a strong indicator of prosody
    human_probability = (0.7 * pitch_score) + (0.3 * mfcc_score)
    
    # Classification decision
    if human_probability >= 0.5:
        classification = "HUMAN"
        confidence = human_probability
    else:
        classification = "AI_GENERATED"
        confidence = 1.0 - human_probability

    # Generate explanation
    explanation = []
    if classification == "HUMAN":
        if pitch_score > 0.5:
            explanation.append("High pitch variability suggests natural intonation.")
        if mfcc_score > 0.5:
            explanation.append("Spectral dynamics indicate natural coarticulation.")
        if not explanation:
             explanation.append("Overall acoustic features lean towards human patterns.")
    else:
        if pitch_score <= 0.5:
            explanation.append("Low pitch variability suggests monotonic/robotic speech.")
        if mfcc_score <= 0.5:
            explanation.append("Low spectral variance indicates lack of natural acoustic richness.")
        if not explanation:
            explanation.append("Overall acoustic features lean towards synthetic patterns.")
        
    return {
        "classification": classification,
        "confidence": float(round(confidence, 4)),
        "explanation": explanation,
        "details": {
            "human_probability": float(round(human_probability, 4)),
            "pitch_score": float(round(pitch_score, 4)),
            "mfcc_score": float(round(mfcc_score, 4))
        }
    }
