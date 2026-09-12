"""
Phase B1: extract one "emotion vector" per emotion from stimuli.json.

Method (contrastive activation / diff-of-means, same idea as ActAdd /
Contrastive Activation Addition): for each emotion, run its 15 sentences and
the 15 topic-matched neutral baselines through the model, mean-pool the
residual stream over tokens at every layer, then take

    emotion_vector[layer] = mean(emotion activations)[layer] - mean(neutral activations)[layer]

This gives one direction per emotion per layer. B2/B3/B4 (next scripts) will
test which layer's vector actually works before we lock one in for the
decay experiment.

Usage:
    python 02_extract_emotion_vectors.py

Requires stimuli.json in the same folder. Takes ~1-3 min on CPU for
gpt2-medium (105 short sentences, one forward pass each).
"""
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
OUTPUT_PATH = "emotion_vectors.pt"


def get_device():
    """Prefer a GPU if one's actually available, otherwise fall back to CPU.
    Order: CUDA (Nvidia) > MPS (Apple Silicon) > CPU. If MPS throws a weird
    error here, hard-code device="cpu" instead of debugging it."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_all_layer_mean_resid(model, text, n_layers):
    """One forward pass; returns [n_layers, d_model] mean-pooled-over-tokens
    residual stream activations."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("hook_resid_post")
        )
    return torch.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0].mean(dim=0) for L in range(n_layers)]
    )


def main():
    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)

    device = get_device()
    print(f"Using device: {device}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"Loaded {MODEL_NAME}: {n_layers} layers, d_model={d_model}")

    neutral_sents = stimuli["neutral_baseline_stories"]
    print(f"Encoding {len(neutral_sents)} neutral baseline sentences...")
    neutral_acts = torch.stack(
        [get_all_layer_mean_resid(model, s, n_layers) for s in neutral_sents]
    )  # [n_neutral, n_layers, d_model]
    neutral_mean = neutral_acts.mean(dim=0)  # [n_layers, d_model]

    emotion_vectors = {}      # emotion -> [n_layers, d_model]  (the diff vector)
    emotion_raw_acts = {}     # emotion -> [15, n_layers, d_model]  (kept for later variance/geometry checks)
    for emotion, sents in stimuli["emotions"].items():
        print(f"Encoding {len(sents)} '{emotion}' sentences...")
        acts = torch.stack(
            [get_all_layer_mean_resid(model, s, n_layers) for s in sents]
        )  # [15, n_layers, d_model]
        emotion_mean = acts.mean(dim=0)  # [n_layers, d_model]
        diff = emotion_mean - neutral_mean  # [n_layers, d_model]
        emotion_vectors[emotion] = diff
        emotion_raw_acts[emotion] = acts

    torch.save(
        {
            "model_name": MODEL_NAME,
            "n_layers": n_layers,
            "d_model": d_model,
            "emotion_vectors": emotion_vectors,
            "neutral_mean": neutral_mean,
            "emotion_raw_acts": emotion_raw_acts,
        },
        OUTPUT_PATH,
    )
    print(f"\nSaved vectors to {OUTPUT_PATH}")

    mid = n_layers // 2
    print(f"\nSanity numbers - vector norm per emotion at layer {mid} "
          f"(should all be clearly non-zero; roughly similar magnitude is a good sign,\n"
          f"wildly different magnitudes across emotions is worth a second look):")
    for emotion, vec in emotion_vectors.items():
        print(f"  {emotion:10s}: norm={vec[mid].norm().item():.3f}")


if __name__ == "__main__":
    main()
