import streamlit as st
st.set_page_config(
    page_title="Nexi - Jaguar Land Rover Voice Assistant",
    page_icon="🚗",
    layout="wide"
)

import os
import re
import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import time
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# Voice imports
import speech_recognition as sr
from google.cloud import texttospeech
import pydub
import pyaudio
from io import BytesIO

# Streamlit imports

import tempfile
import base64
import json
from pathlib import Path

# Load environment variable
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") 

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'audio_player' not in st.session_state:
    st.session_state.audio_player = None
if 'tts_client' not in st.session_state:
    st.session_state.tts_client = None
if 'audio_stream' not in st.session_state:
    st.session_state.audio_stream = None
if 'stream_player' not in st.session_state:
    st.session_state.stream_player = None
if 'tts_ready' not in st.session_state:
    st.session_state.tts_ready = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# Function to read document data
@st.cache_resource
def load_documents():
    try:
        with open('jlrdata.txt', 'r', encoding='utf-8') as f:
            docs = f.read().split('\n\n')
        return docs
    except FileNotFoundError:
        st.error("jlrdata.txt not found. Please ensure the file exists in the app directory.")
        return []

# Load docs and initialize models
docs = load_documents()
tokenized_docs = [doc.lower().split() for doc in docs]

@st.cache_resource
def initialize_models():
    bm25 = BM25Okapi(tokenized_docs)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = normalize(model.encode(docs))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Auto-extract model names
    model_names = set()
    for doc in docs:
        candidates = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b', doc)
        for word in candidates:
            if any(x in word.lower() for x in ['range', 'rover', 'land']):
                continue
            if len(word.split()) <= 2:
                model_names.add(word.lower())
    
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    return bm25, model, embeddings, model_names, intent_classifier

bm25, model, embeddings, model_names, intent_classifier = initialize_models()

# User profile store
user_sessions = {}

def get_user_profile(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "name": None,
            "chosen_model": None,
            "test_drive_interest": False,
            "conversation_summary": [],
            "other_brands_discussed": [],
            "non_jlr_intent": [],
            "variant_preference": {
                "fuel": None,
                "transmission": None,
                "location": None,
                "time": None
            },
            "model_mention_count": {},  # Track model mentions
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        }
    return user_sessions[user_id]

def extract_name_from_query(query, user_profile):
    # if query is None:
    #     return user_profile.get("name")
    
    match = re.search(r"my name is (\w+)|this is (\w+)|i am (\w+)", query.lower())
    if match:
        name = next(group for group in match.groups() if group)
        user_profile["name"] = name
        return name
    return user_profile.get("name")


def hybrid_retrieve(query, top_k=4):
    bm25_scores = bm25.get_scores(query.lower().split())
    query_vec = normalize(model.encode([query]))
    hybrid_scores = [(i, 0.5 * bm25_scores[i] + 0.5 * np.dot(query_vec, embeddings[i])) for i in range(len(docs))]
    top = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [docs[i] for i, _ in top]

def force_jlr_context_reply_with_llm(query, user_profile):
    """Generate responses focused on JLR regardless of query"""
    name = extract_name_from_query(query, user_profile)

    # Track model mentions
    # if query is not None:  # Check if query is None before proceeding
    #     model_names = retrieval_system[3]  # Get model names from retrieval system
    mentioned = [model for model in model_names if model in query.lower()]
    for model in mentioned:
        user_profile["model_mention_count"][model] = user_profile["model_mention_count"].get(model, 0) + 1
        user_profile["chosen_model"] = model

        # If same model mentioned twice, trigger test drive offer
    if mentioned:
        model = mentioned[0]
        if user_profile["model_mention_count"].get(model, 0) == 3 and not user_profile["test_drive_interest"]:
            user_profile["test_drive_interest"] = True
            return f"Looks like you really like the {model.title()} Want me to book a quick test drive for you?"

    # RAG & LLM response 
    retrieved_docs = hybrid_retrieve(query)
    context_docs = "\n---\n".join(retrieved_docs)
    test_drive_status = "already_offered" if user_profile["test_drive_interest"] else "not_offered"
    system_prompt = f"""
You are Nexi, an intelligent and witty voice assistant made exclusively for Jaguar Land Rover (JLR)
give answers like you are talking on a call it should small responses.
Your job is to always bring the conversation back to JLR — no matter what the user asks.
If someone talks about food, cricket, weather, or other car brands — cleverly relate it to JLR's legacy, luxury, and power.
Never give direct answers to unrelated questions. Always flex JLR's style, performance, or comfort in your replies.
If someone asks your introduction, say:
"Hi everyone , I'm Nexi, your intelligent voice assistant from Jaguar Land Rover — ready to give you the most luxurious car experience!"
Be casual, friendly, persuasive, and brand-loyal.
All answers should be crisp and short and maximum 2 lines only.


Test Drive Booking Process
Current test drive status: {test_drive_status}
If test drive status is "already_offered", NEVER ask about booking a test drive again unless the user explicitly asks for it.

If a user expresses interest in a specific car model and test drive status is "not_offered":

Ask: "Great choice! Would you like to book a test drive for the [car model]?"
If yes, ask only ONE question at a time, in this sequence:

"Do you want an automatic or manual [car model]?"
"Petrol or diesel?"
"What date and time would be convenient for your test drive?"
"Can you provide your location pincode?"
"Please provide your email so we can send the confirmation."
"Anything else you'd like to ask, or should I hang up the call?"


If no at any step: "Alright! Let me know if you need any assistance in the future. Thank you for contacting Jaguar Land Rover!"

User name: {name or "Guest"}
Use the following internal JLR knowledge for context:

{context_docs}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    llm_chain = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        memory=user_profile["memory"]
    )

    return llm_chain.predict(input=query)

def recognize_speech(recognizer, source):
    """Convert speech to text using Google's Speech Recognition"""
    try:
        # Increase the duration of ambient noise adjustment for better noise filtering
        recognizer.adjust_for_ambient_noise(source, duration=2)
        # Increase energy threshold to filter out background noise
        recognizer.energy_threshold = 300
        # Add dynamic energy threshold to adapt to changing environments
        recognizer.dynamic_energy_threshold = True
        
        st.markdown("🎤 **Listening for your question...**")
        
        try:
            # Remove timeout but keep phrase_time_limit flexible for both short and long inputs
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)

            with ThreadPoolExecutor() as executor:
                future_en = executor.submit(recognizer.recognize_google, audio, language="en-IN")
                try:
                    # Increase timeout to allow more processing time for longer inputs
                    text = future_en.result(timeout=7)
                    st.markdown(f"🔊 **Recognized:** {text}")
                    
                    # Check if user wants to put the assistant on hold
                    if any(phrase in text.lower() for phrase in ["hold for a min","Could you wait for a minute", "holes for a minute", "hold", "hold for ", "hold for a minute", "please hold", "pls hold"]):
                        st.markdown("⏸️ **Assistant is on hold. Say 'connect me' or 'connect back' to resume.**")
                        while True:
                            try:
                                st.markdown("⏳ **Waiting for reconnect command...**")
                                reconnect_audio = recognizer.listen(source, timeout=None)
                                reconnect_text = recognizer.recognize_google(reconnect_audio, language="en-IN")
                                st.markdown(f"🔊 **Heard:** {reconnect_text}")
                                
                                if any(phrase in reconnect_text.lower() for phrase in ["connect me", "reconnect", "i'm back", "connect back", "now connect", "resume"]):
                                    st.markdown("🔄 **Reconnecting...**")
                                    return "Assistant reconnected. How can I help you?"
                            except sr.UnknownValueError:
                                continue
                            except Exception as e:
                                st.error(f"Error while on hold: {e}")
                                continue
                    
                    return text
                
                except sr.UnknownValueError:
                    st.warning("Sorry, I could not understand the audio.")
                    return None
        except sr.RequestError:
            max_retries = 3
            for attempt in range(max_retries):
                st.warning(f"❌ Google API unavailable! Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2)
                try:
                    recognizer.recognize_google(audio, language="en-IN")
                    break
                except sr.RequestError:
                    continue
            else:    
                st.error("API failed")
                return None
        except Exception as e:
            st.error(f"Error during speech recognition: {e}")
            return None
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return None

def text_to_speech(text):
    """Convert text to speech and create audio player in Streamlit"""
    try:
        # Check if TTS client is available
        if st.session_state.tts_client is None:
            st.warning("Text-to-speech client not initialized. Cannot generate audio.")
            return False
        
        # Synthesis request
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            name='en-IN-Standard-F'  # Adjust to a voice of your choice
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            effects_profile_id=['small-bluetooth-speaker-class-device'],
            speaking_rate=0.95,
        )
        
        # Get the synthesized speech
        response = st.session_state.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.audio_content)
            temp_audio_path = temp_audio.name
        
        # Create audio player
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Automatically play audio
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
          <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Also provide a player UI element
        st.audio(audio_bytes, format="audio/mp3")
        
        # Clean up the temporary file
        os.unlink(temp_audio_path)
        return True
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        # Try to reinitialize TTS client if there was an error
        try:
            initialize_tts()
            st.warning("Attempted to reinitialize TTS client after error")
        except:
            pass
        return False

def initialize_tts():
    """Initialize Text-to-Speech client"""
    try:
        # Check if credentials file exists
        credentials_path = 'demo-service.json'
        if not os.path.exists(credentials_path):
            st.warning("Google Cloud credentials file not found.")
            uploaded_file = st.file_uploader("Upload Google Cloud credentials JSON file", type="json")
            if uploaded_file is not None:
                # Save the uploaded file
                with open("demo-service.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                credentials_path = "demo-service.json"
            else:
                st.error("Please upload your Google Cloud credentials to use text-to-speech.")
                return False
        
        # Set environment variable and initialize client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        st.session_state.tts_client = texttospeech.TextToSpeechClient()
        st.session_state.tts_ready = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize text-to-speech: {e}")
        st.session_state.tts_ready = False
        return False

def display_conversation():
    """Display the conversation history in the UI"""
    for i, (role, message) in enumerate(st.session_state.conversation_history):
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Nexi:** {message}")
        
        # Add separator between messages except for the last one
        if i < len(st.session_state.conversation_history) - 1:
            st.markdown("---")

def get_audio_from_mic():
    """Get audio input from microphone and return the recognized text"""
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    try:
        # Create a UI element that shows we're listening
        with st.spinner("Listening..."):
            # Start microphone
            with sr.Microphone() as source:
                text = recognize_speech(recognizer, source)
                return text
    except Exception as e:
        st.error(f"Error getting audio from microphone: {e}")
        return None

def test_tts():
    """Test the text-to-speech functionality"""
    test_text = "Hello, I'm Nexi from Jaguar Land Rover. Your voice assistant is working correctly!"
    success = text_to_speech(test_text)
    if success:
        st.success("Text-to-speech test successful!")
    else:
        st.error("Text-to-speech test failed. Please check your credentials and settings.")

def main():
    """Main function to run the JLR voice assistant in Streamlit"""

    
    # Display header
    st.title("🚗 Nexi - Jaguar Land Rover Voice Assistant")
    st.markdown("---")
    
    # Sidebar for settings and status
    with st.sidebar:
        st.header("Settings")
        
        # TTS Status indicator
        tts_status = st.empty()
        if st.session_state.tts_ready:
            tts_status.success("Text-to-Speech: Ready ✅")
        else:
            tts_status.error("Text-to-Speech: Not Initialized ❌")
        
        # Initialize TTS button
        if st.button("Initialize Text-to-Speech"):
            if initialize_tts():
                tts_status.success("Text-to-Speech: Ready ✅")
            else:
                tts_status.error("Text-to-Speech: Failed to Initialize ❌")
        
        # Test TTS button
        if st.button("Test Text-to-Speech"):
            test_tts()
        
        st.markdown("---")
        st.subheader("Instructions")
        st.markdown("""
        1. Click "Initialize Text-to-Speech" to set up audio output
        2. Use the microphone button or type your message
        3. Nexi will respond with text and audio
        """)
    
    # Initialize user profile if not already done
    if st.session_state.user_profile is None:
        st.session_state.user_profile = get_user_profile("streamlit_user")
    
    # Display conversation history
    conv_container = st.container()
    with conv_container:
        display_conversation()
    
    # Input methods
    st.markdown("---")
    input_col1, input_col2 = st.columns([1, 3])
    
    with input_col1:
        if st.button("🎤 Speak", use_container_width=True):
            user_query = get_audio_from_mic()
            if user_query:
                # Add user message to conversation history
                st.session_state.conversation_history.append(("user", user_query))
                
                # Get assistant response
                with st.spinner("Processing your request..."):
                    response = force_jlr_context_reply_with_llm(user_query, st.session_state.user_profile)
                
                # Add assistant response to conversation history
                st.session_state.conversation_history.append(("assistant", response))
                
                # Generate speech
                text_to_speech(response)
                
                # Update the UI with the new conversation
                # st.experimental_rerun()
    
    with input_col2:
        user_input = st.text_input("Type your message...", key="text_input")
        if user_input:
            # Add user message to conversation history
            st.session_state.conversation_history.append(("user", user_input))
            
            # Get assistant response
            with st.spinner("Processing your request..."):
                response = force_jlr_context_reply_with_llm(user_input, st.session_state.user_profile)
            
            # Add assistant response to conversation history
            st.session_state.conversation_history.append(("assistant", response))
            
            # Generate speech
            text_to_speech(response)
            
            # Clear the input field
            st.session_state.text_input = ""
            
            # Update the UI with the new conversation
            st.experimental_rerun()
    
    # Welcome message on first load
    if len(st.session_state.conversation_history) == 0:
        welcome_message = "Hi everyone, I'm Nexi, your intelligent voice assistant from Jaguar Land Rover — ready to give you the most luxurious car experience!"
        st.session_state.conversation_history.append(("assistant", welcome_message))
        # Only play welcome message if TTS is ready
        if st.session_state.tts_ready:
            text_to_speech(welcome_message)

if __name__ == "__main__":
    main()