#!/usr/bin/env python3
import os
import io
from dotenv import load_dotenv
import gradio as gr
from elevenlabs.client import ElevenLabs
import ollama
import openai
import tempfile

# Configuration
load_dotenv(override=True)
api_key = os.getenv('ELEVENLABS_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set.\
                     Did you remember to add it in the .env file?")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.\
                      Did you remember to add it in the .env file?")

client = ElevenLabs(api_key=api_key)
initial_system_message = os.getenv('initial_system_message')
elevenlabs_voice = os.getenv('elevenlabs_voice')


def format_history(msg: str, history: list[list[str, str]],
                   system_prompt: str):
    """
    Format chat history for display.
    """
    chat_history = [{"role": "system", "content": system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history


def generate_response(msg: str, history: list[list[str, str]],
                      system_prompt: str):
    """
    Generate a response using the Llama-2 model via Ollama.
    """
    chat_history = format_history(msg, history, system_prompt)
    response = ollama.chat(model='llama2-uncensored', stream=True,
                           messages=chat_history)
    message = ""
    for partial_resp in response:
        if partial_resp["message"]["role"] == "assistant":
            message += partial_resp["message"]["content"]
    return message


def transcribe_audio(audio):
    """
    Transcribe audio using OpenAI's Whisper model.
    """
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=".wav") as temp_audio:
        temp_audio.write(audio.read())
        temp_audio_path = temp_audio.name

    with open(temp_audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    os.unlink(temp_audio_path)
    return transcript["text"]


def respond_to_input(input_data, history: list[list[str, str]],
                     system_prompt: str):
    """
    Respond to user input (text or audio), generate audio using ElevenLabs API,
    and play it.
    """
    global messages

    if isinstance(input_data, str):  # Text input
        text = input_data
    else:  # Audio input
        text = transcribe_audio(input_data)

    messages.append(f"\nUser: {text}")
    # Generate response using Llama-2 model
    response = generate_response(text, history, system_prompt)
    messages.append(f"\nAssistant: {response}")

    # Use ElevenLabs API to convert the response to audio
    audio = client.generate(text=response)

    # Play the audio using elevenlabs play function
    from elevenlabs import play
    play(audio)

    return response


# Initialize messages with the system prompt
messages = [initial_system_message]

# Gradio Interface setup
iface = gr.Interface(
    fn=respond_to_input,
    inputs=[
        gr.inputs.Textbox(lines=2, label="Text Input"),
        gr.inputs.Audio(source="microphone", type="filepath",
                        label="Voice Input"),
        "state",
        gr.inputs.Textbox(initial_system_message, label="System Prompt")
    ],
    outputs=[
        gr.outputs.Textbox(label="Response"),
    ],
    title="LLama-2 (7B) Chatbot with Voice Chat using 'Ollama'",
    description="Feel free to ask any question via text or voice.",
    theme="soft",
)

iface.launch()
