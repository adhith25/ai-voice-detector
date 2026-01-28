import streamlit as st
import requests
import base64
import json

# Configuration
API_URL = "http://localhost:8000/detect-voice"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Page Config
st.set_page_config(
    page_title="AI Voice Detector",
    page_icon="🎙️",
    layout="centered"
)

# Header
st.title("🎙️ AI Voice Detector")
st.markdown("Upload an audio file to check if it's **Human** or **AI-Generated**.")

# 1. Upload Section
st.header("1. Upload Audio")
uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

# 2. Configuration Section
st.header("2. Configuration")
selected_language = st.selectbox("Select Language", SUPPORTED_LANGUAGES)

# 3. Analysis Section
st.header("3. Analysis")

if st.button("Analyze Voice", type="primary"):
    if uploaded_file is not None:
        # Show loading spinner
        with st.spinner("Analyzing audio features..."):
            try:
                # Convert to Base64
                bytes_data = uploaded_file.getvalue()
                base64_audio = base64.b64encode(bytes_data).decode('utf-8')
                
                # Prepare payload
                payload = {
                    "audio_base64": base64_audio,
                    "language": selected_language
                }
                
                # Send Request
                response = requests.post(API_URL, json=payload)
                
                # Handle Response
                if response.status_code == 200:
                    result = response.json()
                    
                    # 4. Result Section
                    st.divider()
                    st.header("4. Results")
                    
                    # Classification & Confidence
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        classification = result.get("classification", "Unknown")
                        if classification == "HUMAN":
                            st.success(f"### {classification}")
                        else:
                            st.error(f"### {classification}")
                        st.caption("Classification")
                        
                    with col2:
                        confidence = result.get("confidence", 0.0)
                        st.metric("Confidence Score", f"{confidence:.1%}")
                    
                    # Explanation
                    st.subheader("Explanation")
                    explanations = result.get("explanation", [])
                    if explanations:
                        for exp in explanations:
                            st.info(f"• {exp}")
                    else:
                        st.write("No specific explanation provided.")
                        
                    # Raw JSON (Expandable)
                    with st.expander("View Raw JSON Response"):
                        st.json(result)
                        
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Is it running?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload an MP3 file first.")

# Footer
st.divider()
st.caption("AI Voice Detector System | Powered by Librosa & FastAPI")
