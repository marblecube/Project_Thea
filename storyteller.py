#!/usr/bin/env python3

import os
import time
from dotenv import load_dotenv
import gradio as gr
from elevenlabs.client import ElevenLabs
import ollama
from openai import OpenAI

# Configuration
load_dotenv(override=True)
api_key = os.getenv('ELEVENLABS_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

if not api_key or not openai_api_key:
    raise ValueError("ELEVENLABS_API_KEY or OPENAI_API_KEY environment variable not set. Did you remember to add them in the .env file?")

client = ElevenLabs(api_key=api_key)
openai_client = OpenAI()

initial_system_message = os.getenv('INITIAL_SYSTEM_MESSAGE')
elevenlabs_voice = "FVQMzxJGPUBtfz1Azdoy"  # Change this to the desired voice ID

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

def process_audio(audio):
    print(f"Processing audio: {audio}")
    if os.path.exists(audio):
        try:
            with open(audio, "rb") as audio_file:
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

def submit_audio(audio, history):
    transcription = process_audio(audio)
    if "Error" in transcription:
        return history, "Error processing audio."
    
    response = respond_to_text(transcription, history)
    history.append((transcription, response))
    return history, ""

def submit_message(text, history):
    if text.strip() == "":
        return history, ""
    response = respond_to_text(text, history)
    history.append((text, response))
    return history, ""

messages = [initial_system_message]

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
