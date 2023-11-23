import openai
import argparse
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text


def create_text(input_text, system_prompt, temperature):
    openai.api_base = "http://localhost:1234/v1"
    openai.api_version = "2023-05-15"
    openai.api_key = ""

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": input_text},
    ]
    
    response = openai.ChatCompletion.create(
        model="local-model",
        messages=messages,
        temperature=temperature,
    )
    message = response.choices[0].message['content']
    
    return message


def get_text(message, hps):
    text_norm = text_to_sequence(message, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(input_text, system_prompt, temperature, speaker, speed, noise_scale, noise_scale_w):
        speaker_id = speaker_ids[speaker]
        
        message = create_text(input_text, system_prompt, temperature)
        
        stn_tst = get_text(message, hps)  
        
        with no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).cuda()
            sid = LongTensor([speaker_id]).cuda()
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return message, (hps.data.sampling_rate, audio)

    return tts_fn


def create_to_phoneme_fn(hps):
    def to_phoneme_fn(text):
        return _clean_text(text, hps.data.text_cleaners) if text != "" else ""

    return to_phoneme_fn

css = """
        #advanced-btn {
            color: white;
            border-color: black;
            background: black;
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 24px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="path to config file")
    parser.add_argument("--model_path", required=True, help="path to model file")
    args = parser.parse_args()

    models_tts = []
    name = ''
    example = '나랑 끝말잇기할래? 내가 먼저할게, 오리온!'
    config_path = args.config_path
    model_path = args.model_path
    hps = utils.get_hparams_from_file(config_path)
    # model 변수를 정의합니다.
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    utils.load_checkpoint(model_path, model, None)
    model.eval()

    # speaker_ids 변수를 정의합니다.
    speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
    speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]

    # speaker_ids 변수를 create_tts_fn 함수에 전달합니다.
    tts_fn = create_tts_fn(model, hps, speaker_ids)

    models_tts.append((name, speakers, example,
                        hps.symbols, create_tts_fn(model, hps, speaker_ids),
                        create_to_phoneme_fn(hps)))

    app = gr.Blocks(css=css)
    
    i = 0  # i 변수를 정의합니다.
    p = 0
    promptExample = " You are 일리야's best friend. Please give a short answer of one or less paragraph in a friendly tone using Korean."

    with app:
        gr.Markdown("VITS-LMStudio Chat demo v0.01")
        with gr.Tabs():
            with gr.TabItem("Setting"):
                 gr.Markdown(f"## LLM Setting")
                 system_prompt = gr.TextArea(label="System prompt: ## Example: You are {user name}'s best friend. Please give a short answer of one or less paragraph in a friendly tone using Korean.", 
                                            elem_id=f"prompt-input{p}", value=promptExample)
                 temperature = gr.Slider(label="Temperature", value=0.9, minimum=0, maximum=2, step=0.1)
                 
                 gr.Markdown(f"## VITS Setting")
                 tts_input2 = gr.Dropdown(label="Speaker", choices=speakers,
                                                        type="index", value=speakers[0])
                 tts_input3 = gr.Slider(label="Speed", value=1, minimum=0.1, maximum=2, step=0.05)
                 noise_scale_slider = gr.Slider(label="Noise-scale (defaults = 0.667)", value=0.667, minimum=0, maximum=1, step=0.01)
                 noise_scale_w_slider = gr.Slider(label="Noise-width (defaults = 0.8)", value=0.8, minimum=0, maximum=2, step=0.05)
                 
            with gr.TabItem(f"VITS-LLM"):
                 gr.Markdown(f"## VITS-LLM")
                 with gr.Column():
                      gr.Markdown()
                      input_text = gr.TextArea(label="Chat: Input text", value=example,
                                                elem_id=f"tts-input{i}")
                    
                      tts_submit = gr.Button("Generate", variant="primary")
                      message = gr.Textbox(label="Output Message")
                      tts_output2 = gr.Audio(label="Output Audio")
                    
                      tts_submit.click(tts_fn, [input_text, system_prompt, temperature, tts_input2, tts_input3, noise_scale_slider, noise_scale_w_slider], [message, tts_output2])
              

        gr.Markdown(
            "Originate from \n\n"
            "- [https://github.com/ilya-scherzo]\n\n"                
        )
    app.queue(concurrency_count=3).launch(share=True)

if __name__ == "__main__":
    main()