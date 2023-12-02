# Code adapted from vits-tts-inference (https://github.com/ilya-scherzo/vitsChats)
# For usage instructions, please refer to the link above.

import json
import os
import torch
from torch import no_grad, LongTensor
from models import SynthesizerTrn
from text import text_to_sequence
import openai
import commons
import utils
import streamlit as st


# Function to set OpenAI LLM model and OpenAI API key parameters
def set_params(model, oai_key):
    os.environ["MODEL"] = model
    os.environ["OPENAI_API_KEY"] = oai_key
    return model, oai_key


# Function to load the path information for the VITS model configuration and checkpoint files
def load_model_info():
    with open('model_path.json', 'r') as f:
        return json.load(f)


# Function to initialize the VITS model with provided hyperparameters and model path
def initialize_model(hps, model_path):
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).cuda()
    utils.load_checkpoint(model_path, model, None)
    model.eval()
    return model


# Function to retrieve speaker IDs and names from the provided hyperparameters
def get_speaker_ids_and_speakers(hps):
    speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
    speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]
    return speaker_ids, speakers


# Function to generate a chat message via OpenAI API using the provided input and model
def get_chat_message(input_text, system_prompt, temperature, model_name):
    if model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"]:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_base = "http://localhost:1234/v1"
        openai.api_version = "2023-05-15"
        openai.api_key = ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message['content']


# Function to normalize the chat message text
def get_normalized_text(message, hps):
    text_norm = text_to_sequence(message, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


# Function to generate Text-to-Speech (TTS) using the VITS model
def generate_tts(
        input_text, system_prompt, temperature, speaker, speed, noise_scale, noise_scale_w, hps, speakers, speaker_ids,
        model, model_name
):
    message = get_chat_message(input_text, system_prompt, temperature, model_name)
    normalized_text = get_normalized_text(message, hps)
    speaker_index = speakers.index(speaker)
    speaker_id = speaker_ids[speaker_index]
    with no_grad():
        x_tst = normalized_text.cuda().unsqueeze(0)
        x_tst_lengths = LongTensor([normalized_text.size(0)]).cuda()
        sid = LongTensor([speaker_id]).cuda()
        audio = model.infer(
            x_tst, 
            x_tst_lengths, 
            sid=sid, 
            noise_scale=noise_scale, 
            noise_scale_w=noise_scale_w, 
            length_scale=1.0 / speed
        )[0][0, 0].data.cpu().float().numpy()
    del normalized_text, x_tst, x_tst_lengths, sid
    return message, (hps.data.sampling_rate, audio)


# Main function to run the Streamlit chat-based user interface application
def main():
    st.set_page_config(page_title="VITS-Chat", page_icon=None, layout='centered', initial_sidebar_state='auto')

    model_info = load_model_info()
    config_path = model_info['config_path']
    model_path = model_info['model_path']
    hps = utils.get_hparams_from_file(config_path)
    model = initialize_model(hps, model_path)
    speaker_ids, speakers = get_speaker_ids_and_speakers(hps)

    st.sidebar.title("OpenAI API Configurator")

    model_name = st.sidebar.selectbox(
        label="Model",
        options=[
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-1106",
            "Input Model name",
            "Local inference (LM studio v0.2.8)",
        ],
        key="model_select"
    )
    if model_name == "Input Model name":
        model_name = st.sidebar.text_input("Enter the model name:", key="model_name_input")
    oai_key = st.sidebar.text_input("OpenAI API Key:", type='password')

    if st.sidebar.button('Set Parameters'):
        model_name, oai_key = set_params(model_name, oai_key)
        st.sidebar.success("Parameters set successfully!")

    st.sidebar.title("Settings")

    system_prompt = st.sidebar.text_area(label="LLM: System prompt", value="Generate short, conversational answers to given questions using Korean.")
    temperature = st.sidebar.slider(label="Temperature", value=0.9, min_value=0.0, max_value=2.0, step=0.1)
    speaker = st.sidebar.selectbox(label="VITS: Speaker", options=speakers, index=0)
    speed = st.sidebar.slider(label="Speed", value=1.0, min_value=0.1, max_value=2.0, step=0.05)
    noise_scale = st.sidebar.slider(label="Noise-scale (defaults = 0.667)", value=0.667, min_value=0.0, max_value=1.0, step=0.01)
    noise_scale_w = st.sidebar.slider(label="Noise-width (defaults = 0.8)", value=0.8, min_value=0.0, max_value=2.0, step=0.05)

    st.sidebar.markdown("🗨️ VITS-Chat: For a more natural voice TTS conversion and chat...")

    st.title("🗨️ VITS-Chat")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if input_text := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.write(input_text)

    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                message, audio_tuple = generate_tts(
                    input_text, 
                    system_prompt, 
                    temperature, 
                    speaker, 
                    speed, 
                    noise_scale,
                    noise_scale_w, 
                    hps, 
                    speakers, 
                    speaker_ids, 
                    model, 
                    model_name
                )
                st.write(message) 
        st.session_state.messages.append({"role": "assistant", "content": message})
        sampling_rate = audio_tuple[0]
        audio_data = audio_tuple[1]
        st.audio(audio_data, format='audio/wav', sample_rate=sampling_rate)
        st.markdown("<script>document.querySelector('audio').play()</script>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
