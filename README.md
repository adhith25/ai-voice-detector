# AI Voice Detector

## Problem Statement
The rapid advancement of Generative AI has made it easier than ever to synthesize realistic human voices. While this technology has positive applications, it also poses significant risks, such as deepfakes, voice phishing (vishing), and misinformation. Distinguishing between authentic human speech and AI-generated audio is becoming a critical challenge for security and trust.

## Solution Overview
The **AI Voice Detector** is a robust, lightweight system designed to classify audio samples as either "Human" or "AI-Generated." Unlike black-box solutions, this system relies on transparent acoustic feature analysis—examining pitch variability, spectral flatness, and MFCC (Mel-frequency cepstral coefficients) variance—to make decisions.

The solution is language-agnostic but explicitly validates support for **Tamil, English, Hindi, Malayalam, and Telugu**, making it highly relevant for diverse linguistic contexts. It provides not just a classification, but also a confidence score and explainable reasons for its decision.

## Key Features
- **REST API Architecture**: Built on high-performance FastAPI for easy integration.
- **Base64 Input**: Accepts Base64-encoded MP3 strings, simplifying client-side uploads.
- **Multi-Language Support**: Explicit validation for Tamil, English, Hindi, Malayalam, and Telugu.
- **Explainable AI**: Returns a list of acoustic reasons (e.g., "Low pitch variability") alongside the result.
- **Confidence Scoring**: Provides a granular confidence score (0.0 - 1.0).
- **Lightweight**: Runs entirely on CPU with no heavy GPU dependencies.
- **Open Source Stack**: Built using standard Python libraries (Librosa, Pydub, FastAPI).

## System Architecture
The data flow follows a streamlined processing pipeline:

1.  **Client**: Sends a POST request with Base64-encoded audio and language metadata.
2.  **API Layer**: FastAPI validates the request schema and language support.
3.  **Audio Processing**: Decodes Base64 to MP3, converts to WAV, and validates duration/silence.
4.  **Feature Extraction**: Extracts key acoustic features (MFCC, Pitch, Spectral Flatness, RMS) using Librosa.
5.  **Classification Engine**: Applies heuristic analysis to determine "human-ness" vs. "synthetic-ness."
6.  **Response**: Returns a JSON object with the classification, confidence score, and explanation.

## Tech Stack
- **Backend**: FastAPI, Python 3.x
- **Frontend**: Streamlit (for interactive demo)
- **Audio Processing**: Librosa, Pydub, FFmpeg
- **Computation**: NumPy
- **Testing**: Pytest

## API Details

### Endpoint
`POST /detect-voice`

### Request Example
```json
{
  "audio_base64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAA...",
  "language": "English"
}
```

### Response Example
```json
{
  "classification": "AI_GENERATED",
  "confidence": 0.85,
  "explanation": [
    "Low pitch variability suggests monotonic/robotic speech.",
    "Low spectral variance indicates lack of natural acoustic richness."
  ],
  "details": {
    "human_probability": 0.15,
    "pitch_score": 0.1,
    "mfcc_score": 0.2
  }
}
```

## How to Run Locally

### Prerequisites
- Python 3.8+
- FFmpeg installed and added to system PATH

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-voice-detector

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Backend
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://localhost:8000`. You can access the Swagger UI documentation at `http://localhost:8000/docs`.

### Step 4: Run the Frontend (Optional)
Open a new terminal and run:
```bash
streamlit run frontend/app.py
```
The interactive UI will open at `http://localhost:8501`.

## Testing
To verify the system integrity, run the automated test suite:
```bash
python -m pytest
```
This executes unit tests for audio processing, feature extraction, and classification logic.

## Limitations & Future Improvements
This project is an MVP (Minimum Viable Product) demonstrating the feasibility of acoustic analysis for voice detection.
- **Current Limitation**: Uses heuristic thresholds which may need tuning for high-quality clones.
- **Future Improvements**:
    - Train a lightweight ML model (Random Forest/XGBoost) on a labeled dataset.
    - Support real-time streaming audio detection.
    - Containerize the application (Docker) for cloud deployment.
    - Expand language support to global languages.

## Conclusion
The **AI Voice Detector** provides a transparent, explainable, and easy-to-deploy solution for identifying synthetic speech. By focusing on fundamental acoustic properties rather than black-box deep learning, it offers a lightweight and interpretable alternative suitable for immediate integration into security and verification workflows.
