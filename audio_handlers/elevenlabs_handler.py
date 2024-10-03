#!/usr/bin/env python3

import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configuration
load_dotenv(override=True)
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set. Did you remember to add it to the .env file?")

client = ElevenLabs(api_key=elevenlabs_api_key)
elevenlabs_voice = os.getenv('ELEVENLABS_VOICE')

def synthesize_speech(text: str):
    try:
        # If generate returns a generator, convert it to a list or process it
        audio = list(client.generate(text=text, voice=elevenlabs_voice))
        logging.info(f"Audio type: {type(audio)}")  # Log the type for debugging
        return audio[0] if audio else None  # Adjust based on your needs
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None


def play_audio(audio):
    try:
        play(audio)  # Play the synthesized audio
        logging.info("Audio played successfully.")
    except Exception as e:
        logging.error(f"Error playing audio: {e}")
