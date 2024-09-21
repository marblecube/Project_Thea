#!/usr/bin/env python3

import os
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

# Custom CSS for gradient background
css = """
    .gradio-container {
        background: linear-gradient(to top, sienna, darksalmon, \
                     darkolivegreen, teal);
        padding: 20px;
        border-radius: 8px;

h1 {
    color: mediumspringgreen !important;
}

a {
    color: darkseagreen !important;
    text-decoration: none !important;
}
a:hover {
    text-decoration: underline !important;
    color: lightseagreen !important;  /* Optional: Darker red on hover */
}
    }
"""

# Gradio Blocks setup
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Project Thea: Interactive Storyteller")
    gr.Markdown("A choose-your-own-adventure, interactive, relationship-builder. Presented by <a href=\"https://www.firecrackermedia.co\">Firecracker Media</a>.")
    
    # Chatbot interface
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=400,
                avatar_images=["user.jpg", "chatbot.png"]
            )
        with gr.Column(scale=2):
            message_box = gr.Textbox(placeholder="Enter your message here...", show_label=False)
            submit_btn = gr.Button("Send ⬅")
            mic_btn = gr.Button("Activate Mic")  # Placeholder for future voice functionality
    
    # Action when submit is clicked
    def submit_message(text, history):
        response = respond_to_text(text, history)
        history.append((text, response))
        return history, ""

    # Clear action
    def clear_chat():
        return []

    submit_btn.click(submit_message, [message_box, chatbot], [chatbot, message_box])

demo.launch()
