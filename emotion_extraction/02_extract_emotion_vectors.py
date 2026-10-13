"""
Phase B1: extract one "emotion vector" per emotion from stimuli.json.

Method (mean-difference contrastive activation / CAA-style):
  1. Take the residual stream at the LAST token of each sentence (not a
     mean over tokens) - the state after the model has finished reading it.
  2. Average those activations over each emotion's sentences.
  3. Subtract the average over all emotions, removing the direction every
     emotion story shares:

    emotion_vector[e] = mean_i( last_token(emotion_e_i) )
                      - mean_e'( mean_i( last_token(emotion_e'_i) ) )

This gives one direction per emotion per layer. The experiments steer at
CHOSEN_LAYER (fixed at 14, picked earlier by a 6-way linear probe sweep over
the mid/late layers of gpt2-medium). Nothing re-checks the layer, so revisit
it if the stimuli or the model change.

The neutral baseline is still encoded and saved with the raw activations,
along with the uncentered emotion - neutral differences, for reference.

Outputs (written to emotion_extraction/results/raw/):
  - emotion_vectors.pt         (raw activations, means, emotion - neutral vectors)
  - emotion_vectors_final.pt   (centered vectors, used by the experiments)
  - config.json                ({"chosen_layer", "chosen_method", "pooling"})

Usage (from the repo root):
    python emotion_extraction/02_extract_emotion_vectors.py

Requires stimuli.json in the repo root. Takes ~1-3 min on CPU for
gpt2-medium (135 short sentences, one forward pass each).
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
RAW_DIR = "emotion_extraction/results/raw"
RAW_VECTORS_PATH = f"{RAW_DIR}/emotion_vectors.pt"
FINAL_VECTORS_PATH = f"{RAW_DIR}/emotion_vectors_final.pt"
CONFIG_PATH = f"{RAW_DIR}/config.json"
METHOD = "vs_general"  # mean last-token emotion - mean over all emotions
POOLING = "last_token"
CHOSEN_LAYER = 14


def get_device():
    """Prefer a GPU if one's actually available, otherwise fall back to CPU.
    Order: CUDA (Nvidia) > MPS (Apple Silicon) > CPU. If MPS throws a weird
    error here, hard-code device="cpu" instead of debugging it."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_all_layer_last_resid(model, text, n_layers):
    """One forward pass; returns [n_layers, d_model] last-token residual
    stream activations."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("hook_resid_post")
        )
    return torch.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :] for L in range(n_layers)]
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
    print(f"Encoding {len(neutral_sents)} neutral baseline sentences (last token)...")
    neutral_acts = torch.stack(
        [get_all_layer_last_resid(model, s, n_layers) for s in neutral_sents]
    )  # [n_neutral, n_layers, d_model]
    neutral_mean = neutral_acts.mean(dim=0)  # [n_layers, d_model]

    emotion_raw_acts = {}     # emotion -> [n_sentences, n_layers, d_model]
    for emotion, sents in stimuli["emotions"].items():
        print(f"Encoding {len(sents)} '{emotion}' sentences (last token)...")
        emotion_raw_acts[emotion] = torch.stack(
            [get_all_layer_last_resid(model, s, n_layers) for s in sents]
        )  # [n_sentences, n_layers, d_model]

    emotion_means = {e: acts.mean(dim=0) for e, acts in emotion_raw_acts.items()}
    general_mean = torch.stack(list(emotion_means.values())).mean(dim=0)
    vs_neutral = {e: m - neutral_mean for e, m in emotion_means.items()}
    emotion_vectors = {e: m - general_mean for e, m in emotion_means.items()}

    os.makedirs(RAW_DIR, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "n_layers": n_layers,
            "d_model": d_model,
            "method": "mean_diff",
            "pooling": POOLING,
            "emotion_vectors": vs_neutral,
            "general_mean": general_mean,
            "neutral_mean": neutral_mean,
            "neutral_raw_acts": neutral_acts,
            "emotion_raw_acts": emotion_raw_acts,
        },
        RAW_VECTORS_PATH,
    )
    print(f"\nSaved raw activations to {RAW_VECTORS_PATH}")

    torch.save(
        {
            "model_name": MODEL_NAME,
            "n_layers": n_layers,
            "d_model": d_model,
            "emotion_vectors": emotion_vectors,
            "method": METHOD,
            "pooling": POOLING,
            "chosen_layer": CHOSEN_LAYER,
        },
        FINAL_VECTORS_PATH,
    )
    print(f"Saved centered vectors to {FINAL_VECTORS_PATH} (method={METHOD})")

    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "chosen_layer": CHOSEN_LAYER,
            "chosen_method": METHOD,
            "pooling": POOLING,
        }, f, indent=2)
    print(f"Wrote {CONFIG_PATH} (chosen_layer={CHOSEN_LAYER})")

    print(f"\nSanity numbers - vector norm per emotion at layer {CHOSEN_LAYER} "
          f"(should all be clearly non-zero; roughly similar magnitude is a good sign,\n"
          f"wildly different magnitudes across emotions is worth a second look):")
    for emotion, vec in emotion_vectors.items():
        print(f"  {emotion:10s}: norm={vec[CHOSEN_LAYER].norm().item():.3f}")


if __name__ == "__main__":
    main()
