#!/usr/bin/env python3

import os
import io
from dotenv import load_dotenv

import gradio as gr
from elevenlabs.client import ElevenLabs
import ollama

# Configuration
load_dotenv(override=True)
api_key = os.getenv('ELEVENLABS_API_KEY')

if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set. Did \
                      you remember to add it in the .env file?")

client = ElevenLabs(api_key=api_key)

initial_system_message = "You are a friendly conversationalist."
elevenlabs_voice = (
    "FVQMzxJGPUBtfz1Azdoy"   # Change this to the desired voice name
)

def format_history(msg: str, history: list[list[str, str]]):
    """
    Format chat history for display.
    """
    chat_history = [{"role": "system", "content": initial_system_message}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

def generate_response(msg: str, history: list[list[str, str]]):
    """
    Generate a response using the Llama-2 model via Ollama.
    """
    chat_history = format_history(msg, history)
    response = ollama.chat(model='llama2-uncensored', stream=True, messages=chat_history)
    message = ""
    for partial_resp in response:
        if partial_resp["message"]["role"] == "assistant":
            message += partial_resp["message"]["content"]
    return message

def respond_to_text(text: str, history: list[list[str, str]]):
    """
    Respond to user text input, generate audio using ElevenLabs API, and play it.
    """
    global messages
    messages.append(f"\nUser: {text}")
    # Generate response using Llama-2 model
    response = generate_response(text, history)
    messages.append(f"\nAssistant: {response}")
    # Use ElevenLabs API to convert the response to audio
    audio = client.generate(text=response)
    # Play the audio using ElevenLabs play function
    from elevenlabs import play
    play(audio)
    return response

# Initialize messages with the system prompt
messages = [initial_system_message]

# Gradio ChatInterface setup
chatbot = gr.ChatInterface(
    respond_to_text,
    chatbot=gr.Chatbot(
        avatar_images=["user.jpg", "chatbot.png"],
        height="64vh"
    ),
    title="LLama-2 (7B) Chatbot using 'Ollama'",
    description="Feel free to ask any question.",
    theme="soft",
    submit_btn="⬅ Send",
    retry_btn="🔄 Regenerate Response",
    undo_btn="↩ Delete Previous",
    clear_btn="🗑️ Clear Chat"
)

chatbot.launch()
