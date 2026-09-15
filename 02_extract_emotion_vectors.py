"""
Phase B1: extract one "emotion vector" per emotion from stimuli.json.

Method (topic-paired contrastive activation / CAA-style):
  1. Take the residual stream at the LAST token of each sentence (not a
     mean over tokens) - the state after the model has finished reading it.
  2. For each of the 15 topic-matched scenarios, subtract the matched
     neutral baseline's last-token activation from the emotion sentence's:
         diff_i = last_token(emotion_i) - last_token(neutral_i)
  3. Average those 15 diffs into one emotion vector per layer.

That holds topic/scenario fixed, so the direction is closer to "this emotion
only".

    emotion_vector[layer] = mean_i( last_token(emotion_i)[layer]
                                  - last_token(neutral_i)[layer] )

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
METHOD = "topic_paired"  # last-token emotion_i - neutral_i, averaged over topics


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
    )  # [n_topics, n_layers, d_model]
    neutral_mean = neutral_acts.mean(dim=0)  # [n_layers, d_model]

    emotion_raw_acts = {}     # emotion -> [n_topics, n_layers, d_model]
    emotion_vectors = {}      # emotion -> [n_layers, d_model]
    for emotion, sents in stimuli["emotions"].items():
        if len(sents) != len(neutral_sents):
            raise ValueError(
                f"'{emotion}' has {len(sents)} sentences but there are "
                f"{len(neutral_sents)} neutrals; topic pairing requires equal counts."
            )
        print(f"Encoding {len(sents)} '{emotion}' sentences (last token, topic-paired)...")
        acts = torch.stack(
            [get_all_layer_last_resid(model, s, n_layers) for s in sents]
        )  # [n_topics, n_layers, d_model]
        # Explicit per-topic contrast, then average ( == mean(acts) - mean(neutrals) ).
        paired_diffs = acts - neutral_acts  # [n_topics, n_layers, d_model]
        emotion_raw_acts[emotion] = acts
        emotion_vectors[emotion] = paired_diffs.mean(dim=0)  # [n_layers, d_model]

    emotion_means = {e: acts.mean(dim=0) for e, acts in emotion_raw_acts.items()}
    general_mean = torch.stack(list(emotion_means.values())).mean(dim=0)

    torch.save(
        {
            "model_name": MODEL_NAME,
            "n_layers": n_layers,
            "d_model": d_model,
            "method": METHOD,
            "pooling": "last_token",
            "emotion_vectors": emotion_vectors,
            "general_mean": general_mean,
            "neutral_mean": neutral_mean,
            "neutral_raw_acts": neutral_acts,
            "emotion_raw_acts": emotion_raw_acts,
        },
        OUTPUT_PATH,
    )
    print(f"\nSaved vectors to {OUTPUT_PATH} (method={METHOD}, pooling=last_token)")

    mid = n_layers // 2
    print(f"\nSanity numbers - vector norm per emotion at layer {mid} "
          f"(should all be clearly non-zero; roughly similar magnitude is a good sign,\n"
          f"wildly different magnitudes across emotions is worth a second look):")
    for emotion, vec in emotion_vectors.items():
        print(f"  {emotion:10s}: norm={vec[mid].norm().item():.3f}")


if __name__ == "__main__":
    main()
