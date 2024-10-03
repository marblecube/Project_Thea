#!/usr/bin/env python3

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from logging_config import setup_logging

# Set up logging
setup_logging()

# Configuration
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Did you remember to add it to the .env file?")

openai_client = OpenAI()

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI's Whisper model."""
    print(f"Processing audio: {audio_path}")
    if os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as audio_file:
                transcription_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcription_text = transcription_response.text
            print(f"Transcription result: {transcription_text}")
            return transcription_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return "Transcription failed."
    else:
        print("Audio file does not exist.")
        return "Audio file does not exist."

def submit_audio(audio_path: str, history: list) -> tuple:
    """Process audio input and update chat history."""
    transcription = transcribe_audio(audio_path)
    if "failed" in transcription:  # More robust error checking
        return history, "Error processing audio."

    # Call respond_to_text from storyteller.py
    from storyteller import respond_to_text  # Import here to keep dependencies clear
    response = respond_to_text(transcription, history)  # Ensure this is defined in storyteller.py
    history.append((transcription, response))
    return history, ""
