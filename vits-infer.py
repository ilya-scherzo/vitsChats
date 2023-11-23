#gradio webui inference v1.01
#python vits-infer.py --config_path path/to/config.json --model_path path/to/model.pth

import argparse
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, speed, noise_scale, noise_scale_w):
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps)
        with no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).cuda()
            sid = LongTensor([speaker_id]).cuda()
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

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
    name = 'VITS-TTS demo v1.01'
    example = '학습은 잘 마치셨나요? 좋은 결과가 있길 바래요.'
    config_path = args.config_path
    model_path = args.model_path
    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    utils.load_checkpoint(model_path, model, None)
    model.eval()
    speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
    speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]

    models_tts.append((name, speakers, example,
                        hps.symbols, create_tts_fn(model, hps, speaker_ids),
                        create_to_phoneme_fn(hps)))

    app = gr.Blocks(css=css)

    with app:
        gr.Markdown("Gradio VITS-TTS Inference demo v1.01\n\n")
        with gr.Tabs():
            for i, (name, speakers, example, symbols, tts_fn,
                    to_phoneme_fn) in enumerate(models_tts):
                with gr.TabItem(f"VITS-TTS_v1.01"):
                    with gr.Column():
                        gr.Markdown(f"## {name}\n\n")
                        tts_input1 = gr.TextArea(label="Text", value=example,
                                                    elem_id=f"tts-input{i}")
                        tts_input2 = gr.Dropdown(label="Speaker", choices=speakers,
                                                    type="index", value=speakers[0])
                        tts_input3 = gr.Slider(label="Speed", value=1, minimum=0.1, maximum=2, step=0.05)

                        noise_scale_slider = gr.Slider(label="Noise-scale (defaults = 0.667)", value=0.667, minimum=0, maximum=1, step=0.01)
                        noise_scale_w_slider = gr.Slider(label="Noise-width (defaults = 0.8)", value=0.8, minimum=0, maximum=2, step=0.05)

                        tts_submit = gr.Button("Generate", variant="primary")
                        tts_output1 = gr.Textbox(label="Output Message")
                        tts_output2 = gr.Audio(label="Output Audio")
                        tts_submit.click(tts_fn, [tts_input1, tts_input2, tts_input3, noise_scale_slider, noise_scale_w_slider],
                                            [tts_output1, tts_output2])

        gr.Markdown(
            "Originate from \n\n"
            "- [https://github.com/kdrkdrkdr]\n\n"                
        )
    app.queue(concurrency_count=3).launch(share=True)

if __name__ == "__main__":
    main()
