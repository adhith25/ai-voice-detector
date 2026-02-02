from fastapi import FastAPI, HTTPException, status, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, validator, Field, model_validator
from typing import Optional, Dict, Any, List
import base64
import io
import tempfile
import os
import librosa
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import numpy as np
import sys

# Add current directory to sys.path to allow simple imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from features import extract_features
from model import classify_voice

# API Key Security Configuration
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validates the API key from the request header against the environment variable.
    """
    expected_api_key = os.getenv("VOICE_API_KEY")
    
    # If the server hasn't set an API key, we default to denying access for security
    if not expected_api_key:
         # You might want to log this in production
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key not configured on server",
        )
         
    if api_key_header == expected_api_key:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

app = FastAPI(
    title="Voice Classification API",
    description="API for detecting voice classification from audio input",
    version="1.0.0"
)

# Supported languages
SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

# Define the input model
class VoiceDetectionRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64 encoded MP3 audio string")
    audioBase64: Optional[str] = Field(None, description="Alias for audio_base64")
    audio_base64_format: Optional[str] = Field(None, description="Alias for audio_base64")
    language: Optional[str] = Field(None, description="Language code of the audio")

    @model_validator(mode='before')
    @classmethod
    def check_audio_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Identify which keys are present and not None
            present_keys = [k for k in ['audio_base64', 'audioBase64', 'audio_base64_format'] 
                           if k in data and data[k] is not None]
            
            if not present_keys:
                raise ValueError("Missing audio payload. Provide exactly one of: 'audio_base64', 'audioBase64', 'audio_base64_format'.")
            
            if len(present_keys) > 1:
                raise ValueError(f"Ambiguous input. Only one audio field allowed. Found: {', '.join(present_keys)}")
            
            # Normalize to audio_base64
            key = present_keys[0]
            if key != 'audio_base64':
                data['audio_base64'] = data[key]
        return data

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
async def detect_voice(request: VoiceDetectionRequest, api_key: str = Depends(get_api_key)):
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
    
    temp_mp3_path = None
    
    try:
        # 1. Preprocess Base64 string
        base64_str = request.audio_base64.strip()
        
        # Handle Data URI scheme (e.g., "data:audio/mp3;base64,.....")
        if "," in base64_str:
            base64_str = base64_str.split(",")[-1]
            
        # 2. Decode Base64 string to bytes
        try:
            audio_data = base64.b64decode(base64_str, validate=True)
        except Exception:
            # We treat base64 errors as "decoding failed" fallback now per requirement
            raise ValueError("Invalid Base64")

        # 3. Write bytes to temporary MP3 file
        # Use NamedTemporaryFile with delete=False to avoid Windows file lock issues during read
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            temp_mp3.write(audio_data)
            temp_mp3.flush()
            temp_mp3.close()
            temp_mp3_path = temp_mp3.name

        # 4. Load audio using librosa
        try:
            # sr=None preserves native sampling rate, mono=True mixes down to mono
            y, sr = librosa.load(temp_mp3_path, sr=None, mono=True)
        except Exception as e:
            raise ValueError(f"Librosa load failed: {str(e)}")

        # 5. Validate audio length and format
        duration = len(y) / sr
        
        if duration < 0.1:
            raise ValueError("Audio is too short (min 0.1s)")
        if duration > 60:
             raise ValueError("Audio is too long (max 60s)")

        # Check if audio is silent
        if np.max(np.abs(y)) < 0.001:
             raise ValueError("Audio is too silent")
             
        # 6. Extract Features
        try:
            features = extract_features(y, sr)
        except Exception as e:
             raise ValueError(f"Feature extraction failed: {str(e)}")
        
        # 7. Classify
        try:
            result = classify_voice(features)
        except Exception as e:
            raise ValueError(f"Classification failed: {str(e)}")
            
    except Exception as e:
        # GRACEFUL FALLBACK for ANY error in the pipeline
        # Log the error internally but do not expose it to the user
        print(f"Pipeline failed: {e}")
        return VoiceDetectionResponse(
            classification="AI_GENERATED",
            confidence=0.5,
            explanation=["Audio could not be decoded reliably; fallback response returned"],
            details=None
        )
    finally:
        # Clean up temporary file
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            try:
                os.remove(temp_mp3_path)
            except:
                pass

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
