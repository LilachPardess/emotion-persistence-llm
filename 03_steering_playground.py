"""
Phase B2: interactive steering playground for the emotion vectors from B1.

Launches a local Gradio web app where you can:
  - pick a prompt (or one of the neutral baseline stories)
  - pick an emotion vector, a layer, and a steering strength
  - generate a normal ("baseline") continuation and a "steered" one, where
    the emotion vector is added into the residual stream at every generated
    position
  - see a bar chart of how strongly each continuation's own activations
    project onto that emotion vector

This is the manual sanity check for whether a vector actually "is" the
emotion it's supposed to be: if cranking up the "sad" vector doesn't make
generations read sadder, that vector (or that layer) isn't ready for the
decay experiment - go find one that works before moving to Phase C.

Usage:
    pip install gradio
    python 03_steering_playground.py
Then open the local URL it prints (usually http://127.0.0.1:7860).

Requires emotion_vectors.pt (from 02_extract_emotion_vectors.py) and
stimuli.json in the same folder.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
VECTORS_PATH = "emotion_vectors.pt"
STIMULI_PATH = "stimuli.json"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


print("Loading model and vectors - this happens once at startup, then stays warm...")
device = get_device()
print(f"Using device: {device}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.eval()
n_layers = model.cfg.n_layers

saved = torch.load(VECTORS_PATH, map_location=device, weights_only=False)
emotion_vectors = saved["emotion_vectors"]  # dict[str] -> tensor [n_layers, d_model]
EMOTIONS = list(emotion_vectors.keys())

with open(STIMULI_PATH) as f:
    stimuli = json.load(f)
EXAMPLE_PROMPTS = stimuli["neutral_baseline_stories"]


def get_mean_resid_at_layer(text, layer):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
    return cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0)


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def generate_and_compare(prompt, emotion, layer, strength, max_new_tokens, seed):
    layer = int(layer)
    vector = emotion_vectors[emotion][layer].to(device)
    # "strength" is a multiplier of the vector's own natural norm, so
    # strength=1 means "add one native-sized step of this emotion".
    direction = vector / vector.norm()
    add_vec = direction * vector.norm() * strength

    tokens = model.to_tokens(prompt)

    torch.manual_seed(int(seed))
    with torch.no_grad():
        out_base = model.generate(
            tokens, max_new_tokens=int(max_new_tokens), do_sample=True,
            temperature=0.8, stop_at_eos=False, verbose=False,
        )
    baseline_text = model.to_string(out_base[0, tokens.shape[1]:])

    def add_hook(resid, hook):
        return resid + add_vec

    torch.manual_seed(int(seed))
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", add_hook)]):
            out_steered = model.generate(
                tokens, max_new_tokens=int(max_new_tokens), do_sample=True,
                temperature=0.8, stop_at_eos=False, verbose=False,
            )
    steered_text = model.to_string(out_steered[0, tokens.shape[1]:])

    # Measure projection cleanly (hook off) on the full text each continuation produced.
    base_act = get_mean_resid_at_layer(prompt + baseline_text, layer)
    steered_act = get_mean_resid_at_layer(prompt + steered_text, layer)
    base_cos = cosine(base_act, vector)
    steered_cos = cosine(steered_act, vector)

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(["baseline", "steered"], [base_cos, steered_cos],
                   color=["#888888", "#d1495b"])
    ax.set_ylabel(f"cosine sim to '{emotion}' vector")
    ax.set_title(f"layer {layer}, strength {strength}")
    ax.bar_label(bars, fmt="%.3f")
    fig.tight_layout()

    return baseline_text, steered_text, fig


with gr.Blocks(title="Emotion Vector Steering Playground") as demo:
    gr.Markdown(
        "# Emotion Vector Steering Playground\n"
        "Generate a normal continuation and a 'steered' one side by side, and "
        "see whether the vector actually nudges the text toward that emotion. "
        "This is the manual check before trusting a vector for the decay experiment."
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value=EXAMPLE_PROMPTS[0], lines=3)
            gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt, label="Neutral example prompts")
            emotion = gr.Dropdown(choices=EMOTIONS, value=EMOTIONS[0], label="Emotion vector")
            layer = gr.Slider(0, n_layers - 1, value=n_layers // 2, step=1, label="Layer")
            strength = gr.Slider(0, 15, value=5, step=0.5,
                                  label="Steering strength (x native vector norm)")
            max_new_tokens = gr.Slider(10, 80, value=40, step=5, label="Generated tokens")
            seed = gr.Number(value=0, label="Random seed (keep fixed for a fair before/after)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            baseline_out = gr.Textbox(label="Baseline continuation (no steering)", lines=5)
            steered_out = gr.Textbox(label="Steered continuation", lines=5)
            plot_out = gr.Plot(label="Projection onto emotion vector")

    btn.click(
        generate_and_compare,
        inputs=[prompt, emotion, layer, strength, max_new_tokens, seed],
        outputs=[baseline_out, steered_out, plot_out],
    )

if __name__ == "__main__":
    demo.launch()
