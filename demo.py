import gradio
from main import process_gradio_input
import json

with open("tests/config.json") as f:
    config = json.load(f)

examples = []
for k, v in config.items():
    examples.append([f"tests/{k}", v["scale"], v["margin"], 2, False, v["text_scale"]])

gradio.Interface(
    fn=process_gradio_input,
    inputs=[
        gradio.Image(label="Input image"),
        gradio.Number(label="Scale"),
        gradio.Number(label="Margin"),
        gradio.Number(label="Minimum number of words per line"),
        gradio.Number(label="Text size in visualization"),
        gradio.Checkbox(value=False, label="Use Word Beam Search"),
    ],
    outputs=[
        gradio.Textbox(label="Read Text (Raw)"),
        # gradio.Textbox(label="Read Text"),
        gradio.Image(label="Visualization"),
    ],
    examples=examples,
    allow_flagging="never",
    title="Detect and Read Handwritten Words",
    theme=gradio.themes.Monochrome(),
).launch()
