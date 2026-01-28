import librosa
import numpy as np

def extract_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """
    Extracts acoustic features from an audio waveform.

    Features extracted:
    - MFCC mean and variance
    - Pitch (Fundamental Frequency) variance
    - Spectral flatness mean
    - RMS energy variance

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    features = {}

    # 1. MFCC mean and variance
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
    features["mfcc_var"] = np.var(mfcc, axis=1).tolist()

    # 2. Pitch variance (Fundamental Frequency - F0)
    # Using librosa.pyin for pitch estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7')
    )
    # Filter out unvoiced parts (NaNs)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) > 0:
        features["pitch_var"] = float(np.var(f0_clean))
    else:
        features["pitch_var"] = 0.0

    # 3. Spectral flatness mean
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))

    # 4. RMS energy variance
    rms = librosa.feature.rms(y=y)
    features["rms_var"] = float(np.var(rms))

    return features
