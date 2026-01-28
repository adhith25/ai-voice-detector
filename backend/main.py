from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any, List
import base64
import io
import tempfile
import os
import numpy as np
from pydub import AudioSegment
import librosa
import sys

# Add current directory to sys.path to allow simple imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from features import extract_features
from model import classify_voice

app = FastAPI(
    title="Voice Classification API",
    description="API for detecting voice classification from audio input",
    version="1.0.0"
)

# Supported languages
SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

# Define the input model
class VoiceDetectionRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded MP3 audio string")
    language: Optional[str] = Field(None, description="Language code of the audio")

    @validator("language")
    def validate_language(cls, v):
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{v}' is not supported. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
        return v

# Define the output model
class VoiceDetectionResponse(BaseModel):
    classification: str
    confidence: float
    explanation: List[str]
    details: Optional[Dict[str, Any]] = None

@app.post("/detect-voice", response_model=VoiceDetectionResponse)
async def detect_voice(request: VoiceDetectionRequest):
    """
    Detects the classification of the voice from base64 encoded audio.
    
    - **audio_base64**: Base64 encoded audio string (Required). Must be MP3 format.
    - **language**: Language code of the audio (Optional). Supported: Tamil, English, Hindi, Malayalam, Telugu.
    
    Returns:
    - **classification**: The predicted class ('HUMAN' or 'AI_GENERATED')
    - **confidence**: The confidence score of the prediction (0.0 to 1.0)
    - **explanation**: A list of reasons for the classification
    - **details**: Additional details about the classification scores
    """
    
    try:
        # 1. Decode Base64 string
        try:
            audio_data = base64.b64decode(request.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 string")

        # 2. Convert MP3 to WAV using pydub
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            temp_mp3.write(audio_data)
            temp_mp3_path = temp_mp3.name

        wav_path = temp_mp3_path.replace(".mp3", ".wav")
        
        try:
            # Load MP3 and export as WAV
            try:
                sound = AudioSegment.from_mp3(temp_mp3_path)
                sound.export(wav_path, format="wav")
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid audio format. Please ensure input is a valid MP3.")
            
            # 3. Load audio using librosa
            try:
                y, sr = librosa.load(wav_path, sr=None)
            except Exception as e:
                 raise HTTPException(status_code=400, detail="Could not process audio file.")

            # 4. Validate audio length and format
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 0.1:
                raise HTTPException(status_code=400, detail="Audio is too short (min 0.1s)")
            if duration > 60:
                 raise HTTPException(status_code=400, detail="Audio is too long (max 60s)")

            # Check if audio is silent
            if np.max(np.abs(y)) < 0.001:
                 raise HTTPException(status_code=400, detail="Audio is too silent")
                 
            # 5. Extract Features
            # We extract acoustic features (MFCC, pitch, etc.) from the loaded audio waveform
            try:
                features = extract_features(y, sr)
            except Exception as e:
                 raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
            
            # 6. Classify
            # We pass the extracted features to the heuristic model to get classification and explanation
            try:
                result = classify_voice(features)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(temp_mp3_path):
                try:
                    os.remove(temp_mp3_path)
                except:
                    pass
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    return VoiceDetectionResponse(
        classification=result["classification"],
        confidence=result["confidence"],
        explanation=result["explanation"],
        details=result.get("details")
    )

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Voice Classification API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
