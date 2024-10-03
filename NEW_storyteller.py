#!/usr/bin/env python3

import os
import logging
import time
from dotenv import load_dotenv
import gradio as gr
import ollama

from audio_handlers.whisper_handler import transcribe_audio
from audio_handlers.elevenlabs_handler import synthesize_speech, client

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv(override=True)
initial_system_message = os.getenv('INITIAL_SYSTEM_MESSAGE')

def format_history(msg: str, history: list[list[str, str]]):
    chat_history = [{"role": "system", "content": initial_system_message}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

def generate_response(msg: str, history: list[list[str, str]]):
    chat_history = format_history(msg, history)
    retries = 3
    for attempt in range(retries):
        try:
            response = ollama.chat(model='llama2-uncensored', stream=True, messages=chat_history)
            message = ""
            for partial_resp in response:
                if partial_resp["message"]["role"] == "assistant":
                    message += partial_resp["message"]["content"]
            return message
        except Exception as e:
            print(f"Error generating response: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(2)
            else:
                raise Exception("Failed to generate response after retries")

def respond_to_text(text: str, history: list[list[str, str]]):
    global messages
    messages.append(f"\nUser: {text}")
    response = generate_response(text, history)
    messages.append(f"\nAssistant: {response}")
    audio = client.generate(text=response)
    from elevenlabs import play
    play(audio)
    return response

def submit_message(text, history):
    if text.strip() == "":
        return history, ""
    response = respond_to_text(text, history)
    history.append((text, response))
    return history, ""

messages = [initial_system_message]

# Set up Gradio interface
css = """
    .gradio-container {
        background: linear-gradient(to top, sienna, darksalmon, darkolivegreen, teal);
        padding: 20px;
        border-radius: 8px;
    }
    h1 {
        color: mediumspringgreen !important;
    }
    a {
        color: darkseagreen !important;
        text-decoration: none !important;
    }
    a:hover {
        text-decoration: underline !important;
        color: lightseagreen !important;
    }
"""

def submit_message(text, history):
    if text.strip() == "":
        return history, ""
    response = respond_to_text(text, history)
    history.append((text, response))
    return history, ""


# Function to submit audio input
def submit_audio(audio_path, chatbot):
    # Transcribe audio input
    transcribed_text = transcribe_audio(audio_path)
    # Optionally process the transcribed text and generate a response
    response = synthesize_speech(transcribed_text)  # Add logic for response generation
    chatbot.append(("User", transcribed_text))
    chatbot.append(("Bot", response))
    return chatbot, ""


with gr.Blocks(css=css) as demo:
    gr.Markdown("# Project Thea: Interactive Storyteller")
    gr.Markdown("An interactive, relationship-building, choose-your-own-adventure. Presented by <a href=\"https://www.firecrackermedia.co\">Firecracker Media</a>.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=400,
                avatar_images=["user.jpg", "chatbot.png"]
            )
        with gr.Column(scale=2):
            message_box = gr.Textbox(placeholder="Enter your message here...", show_label=False)
            audio_component = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")

    def clear_chat():
        return []

    message_box.submit(submit_message, [message_box, chatbot], [chatbot, message_box])
    audio_component.change(submit_audio, [audio_component, chatbot], [chatbot, message_box])

demo.launch()
