# Modified from vits-tts-inference [https://github.com/ilya-scherzo/vitsChats]
# how to use: visit .................................................here^

import json
import torch
from torch import no_grad, LongTensor
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
import openai
import commons
import utils
import streamlit as st


def load_model_info():
    with open('model_path.json', 'r') as f:
        return json.load(f)


def initialize_model(hps, model_path):
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    utils.load_checkpoint(model_path, model, None)
    model.eval()
    return model


def get_speaker_ids_and_speakers(hps):
    speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
    speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]
    return speaker_ids, speakers


def get_chat_message(input_text, system_prompt, temperature):
    openai.api_base = "http://localhost:1234/v1"
    openai.api_version = "2023-05-15"
    openai.api_key = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]
    response = openai.ChatCompletion.create(
        model="local-model",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message['content']


def get_normalized_text(message, hps):
    text_norm = text_to_sequence(message, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def generate_tts(
    input_text, system_prompt, temperature, speaker, speed, noise_scale, noise_scale_w, hps, speakers, speaker_ids, model
):
    message = get_chat_message(input_text, system_prompt, temperature)
    normalized_text = get_normalized_text(message, hps)
    speaker_index = speakers.index(speaker)
    speaker_id = speaker_ids[speaker_index]
    with no_grad():
        x_tst = normalized_text.cuda().unsqueeze(0)
        x_tst_lengths = LongTensor([normalized_text.size(0)]).cuda()
        sid = LongTensor([speaker_id]).cuda()
        audio = model.infer(
            x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0 / speed
        )[0][0, 0].data.cpu().float().numpy()
    del normalized_text, x_tst, x_tst_lengths, sid
    return message, (hps.data.sampling_rate, audio)


st.set_page_config(page_title="VITS-Chat", page_icon=None, layout='centered', initial_sidebar_state='auto')


def main():
    model_info = load_model_info()
    config_path = model_info['config_path']
    model_path = model_info['model_path']
    hps = utils.get_hparams_from_file(config_path)
    model = initialize_model(hps, model_path)
    speaker_ids, speakers = get_speaker_ids_and_speakers(hps)
    system_prompt = "Generate short, conversational answers to given questions using Korean."
    temperature = 0.9
    speaker = speakers[0]
    speed = 1.0
    noise_scale = 0.667
    noise_scale_w = 0.8

    st.title("🗨️ VITS-Chat")

    page = st.sidebar.selectbox("Choose a page", ["Settings", "Chat"], key="page_select")

    if page == "Settings":
        st.header("Settings")
        system_prompt = st.text_area(label="System prompt: ", value=system_prompt)
        temperature = st.slider(label="LLM: Temperature", value=temperature, min_value=0.0, max_value=2.0, step=0.1)
        speaker = st.selectbox(label="VITS: Speaker", options=speakers, index=speakers.index(speaker))
        speed = st.slider(label="Speed", value=speed, min_value=0.1, max_value=2.0, step=0.05)
        noise_scale = st.slider(label="Noise-scale (defaults = 0.667)", value=noise_scale, min_value=0.0, max_value=1.0,
                                step=0.01)
        noise_scale_w = st.slider(label="Noise-width (defaults = 0.8)", value=noise_scale_w, min_value=0.0, max_value=2.0,
                                  step=0.05)

    elif page == "Chat":
        st.header("Chat")
        input_text = st.chat_input("Say Something, here")
        if input_text:
            message, audio_tuple = generate_tts(input_text, system_prompt, temperature, speaker, speed, noise_scale,
                                                noise_scale_w, hps, speakers, speaker_ids, model)
            sampling_rate = audio_tuple[0]
            audio_data = audio_tuple[1]
            st.text_area(label="Output Message", value=message)
            st.audio(audio_data, format='audio/wav', sample_rate=sampling_rate)
            st.markdown("<script>document.querySelector('audio').play()</script>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
