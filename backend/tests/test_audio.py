from fastapi.testclient import TestClient
from backend.main import app
from pydub.generators import Sine
import io
import base64
import pytest

client = TestClient(app)

def generate_mp3_base64(duration_ms=1000, frequency=440, silent=False):
    if silent:
        sound = Sine(frequency).to_audio_segment(duration=duration_ms).apply_gain(-100)
    else:
        sound = Sine(frequency).to_audio_segment(duration=duration_ms)
    
    buffer = io.BytesIO()
    sound.export(buffer, format="mp3")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def test_detect_voice_valid():
    # Generate 1 second of 440Hz sine wave
    audio_b64 = generate_mp3_base64(duration_ms=1000)
    response = client.post("/detect-voice", json={"audio_base64": audio_b64})
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "confidence" in data

def test_detect_voice_too_short():
    # Generate 50ms of audio
    audio_b64 = generate_mp3_base64(duration_ms=50)
    response = client.post("/detect-voice", json={"audio_base64": audio_b64})
    assert response.status_code == 400
    assert "Audio is too short" in response.json()["detail"]

def test_detect_voice_silent():
    # Generate 1s of silence
    audio_b64 = generate_mp3_base64(duration_ms=1000, silent=True)
    response = client.post("/detect-voice", json={"audio_base64": audio_b64})
    assert response.status_code == 400
    assert "Audio is too silent" in response.json()["detail"]

def test_detect_voice_invalid_base64():
    response = client.post("/detect-voice", json={"audio_base64": "invalid_base64_string"})
    assert response.status_code == 400
    assert "Invalid base64 string" in response.json()["detail"]
