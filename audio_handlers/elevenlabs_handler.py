#!/usr/bin/env python3

import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from logging_config import setup_logging

# Set up logging
setup_logging()

# Configuration
load_dotenv(override=True)
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set. Did you remember to add it to the .env file?")

client = ElevenLabs(api_key=elevenlabs_api_key)
elevenlabs_voice = os.getenv('ELEVENLABS_VOICE')

def synthesize_speech(text: str):
    try:
        # Generate the speech (which returns a generator)
        audio_generator = client.generate(text=text, voice=elevenlabs_voice)
        
        # Collect the audio data from the generator
        audio = b"".join(audio_generator)
        
        # Check if any audio data was collected
        if not audio:
            logging.error("No audio was generated.")
            return None

        # Log the audio type and a portion of the content
        logging.info(f"Audio type after joining: {type(audio)}")
        logging.info(f"Audio content (first 100 bytes): {audio[:100]}")
        
        return audio

    except Exception as e:
        logging.error(f"Error synthesizing speech: {e}")
        return None


def play_audio(audio):
    try:
        play(audio)  # Play the synthesized audio
        logging.info("Audio played successfully.")
    except Exception as e:
        logging.error(f"Error playing audio: {e}")
