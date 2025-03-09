import whisper
from gtts import gTTS
from groq import Groq
import gradio as gr
import os

# Load Whisper model
whisper_model = whisper.load_model("base")

GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

# Function to transcribe audio to text
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result['text']

# Function to interact with Groq API for LLM response
def query_groq(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        model="llama3-8b-8192",  
    )
    return chat_completion.choices[0].message.content

# Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")  
    return "response.mp3"

# Function that integrates all steps: audio input -> transcription -> LLM -> TTS
def chatbot(audio):
    # Step 1: Transcribe the audio to text using Whisper
    transcribed_text = transcribe_audio(audio)

    # Step 2: Get response from Groq's LLM
    llm_response = query_groq(transcribed_text)

    # Step 3: Convert LLM response text to speech
    response_audio = text_to_speech(llm_response)

    # Return both text response and audio file
    return llm_response, response_audio

# Gradio interface
interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Audio(type="filepath"),  # Audio input without 'source' argument
    outputs=[gr.Textbox(label="Response Text"), gr.Audio(label="Response Audio")],  # Text and audio output
    title="Real-Time Voice-to-Voice Chatbot"
)

# Launch the interface
interface.launch()
