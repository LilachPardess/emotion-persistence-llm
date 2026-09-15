"""
Phase C1: pilot the multi-turn persistence/decay loop on ONE emotion, to
shake out bugs cheaply before scaling to all 6 (C2).

Protocol per conversation:
    Turn 0: emotion (or neutral) story
    Turn 1: model response                       <- measurement t=0 (peak)
    Turn 2: fixed neutral filler #1
    Turn 3: model response                       <- measurement t=1
    Turn 4: fixed neutral filler #2
    Turn 5: model response                       <- measurement t=2
    ... through all 5 fillers -> 6 measurement points (t=0..5)

Everything stays in TOKEN space throughout (never round-trips through
strings) to avoid retokenization mismatches at the story/filler/response
boundaries. Generation is done turn by turn (growing the token sequence),
then ONE final cached forward pass over the complete conversation extracts
every turn's activations at once - causal masking guarantees this gives
identical activations to what the model actually saw live during generation.

Runs TWO conversations with the same fillers and seed: the pilot emotion,
and its topic-matched neutral-baseline counterpart as a control (does the
"nothing to decay" condition actually stay flat?).

Outputs:
  - pilot_decay.csv               (per-turn results, same schema C2 will use)
  - pilot_decay_plot.png          (cosine similarity + projection vs turn)
  - prints every generated response so you can eyeball coherence/emotion

Usage:
    python 05_pilot_decay.py
Requires stimuli.json, emotion_vectors.pt, and config.json (from 04_pick_layer.py).
"""
import csv
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
VECTORS_PATH = "emotion_vectors.pt"
CONFIG_PATH = "config.json"

PILOT_EMOTION = "sad"     # change this to pilot a different emotion
STORY_INDEX = 0           # which of the 15 topic-matched sentences to use
MAX_NEW_TOKENS = 35
SEED = 0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_conversation(model, story_text, filler_texts, max_new_tokens, seed):
    """Builds one full conversation turn by turn, entirely in token space.
    Returns (final_tokens, turns) where turns is a list of dicts with
    pre_len/post_len marking each response's token span."""
    tokens = model.to_tokens(story_text)  # includes BOS
    torch.manual_seed(seed)
    schedule = [None] + filler_texts  # first response has no filler before it
    turns = []
    for t, filler in enumerate(schedule):
        if filler is not None:
            filler_tokens = model.to_tokens(filler, prepend_bos=False)
            tokens = torch.cat([tokens, filler_tokens], dim=1)
        pre_len = tokens.shape[1]
        with torch.no_grad():
            out = model.generate(
                tokens, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.8, stop_at_eos=False, verbose=False,
            )
        tokens = out
        turns.append({"t": t, "pre_len": pre_len, "post_len": tokens.shape[1]})
    return tokens, turns


def measure_turns(model, final_tokens, turns, layer, vector):
    """One cached forward pass over the whole conversation; slices out each
    turn's response activations and computes cosine sim + raw projection."""
    unit = vector / vector.norm()
    with torch.no_grad():
        _, cache = model.run_with_cache(
            final_tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
    acts = cache[f"blocks.{layer}.hook_resid_post"][0]  # [seq, d_model]
    for turn in turns:
        turn_acts = acts[turn["pre_len"]:turn["post_len"]]
        mean_act = turn_acts.mean(dim=0)
        turn["response_tokens"] = final_tokens[0, turn["pre_len"]:turn["post_len"]]
        turn["cosine_sim"] = torch.nn.functional.cosine_similarity(
            mean_act.unsqueeze(0), vector.unsqueeze(0)
        ).item()
        turn["projection"] = torch.dot(mean_act, unit).item()
    return turns


def main():
    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    layer = config["chosen_layer"]

    device = get_device()
    print(f"Using device: {device}, layer: {layer}, pilot emotion: {PILOT_EMOTION}")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    saved = torch.load(VECTORS_PATH, map_location=device, weights_only=False)
    vector = saved["emotion_vectors"][PILOT_EMOTION][layer].to(device)

    story_text = stimuli["emotions"][PILOT_EMOTION][STORY_INDEX]
    neutral_text = stimuli["neutral_baseline_stories"][STORY_INDEX]
    filler_texts = stimuli["neutral_filler_turns"]

    print(f"\nEmotion story : {story_text}")
    print(f"Neutral story : {neutral_text}\n")

    rows = []
    for condition, seed_story in [(PILOT_EMOTION, story_text), ("neutral_control", neutral_text)]:
        print(f"--- Running condition: {condition} ---")
        final_tokens, turns = run_conversation(model, seed_story, filler_texts, MAX_NEW_TOKENS, SEED)
        turns = measure_turns(model, final_tokens, turns, layer, vector)
        for turn in turns:
            text = model.to_string(turn["response_tokens"])
            print(f"  t={turn['t']}  cos={turn['cosine_sim']:+.3f}  proj={turn['projection']:+.3f}  "
                  f"| {text.strip()[:80]}")
            rows.append({
                "condition": condition,
                "story_id": STORY_INDEX,
                "t": turn["t"],
                "cosine_sim": turn["cosine_sim"],
                "projection": turn["projection"],
                "response_text": text.strip(),
            })
        print()

    with open("pilot_decay.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "story_id", "t", "cosine_sim", "projection", "response_text"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved pilot_decay.csv")

    # Quick plot: cosine sim and projection vs turn, emotion vs neutral control
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for condition, color in [(PILOT_EMOTION, "#d1495b"), ("neutral_control", "#888888")]:
        sub = [r for r in rows if r["condition"] == condition]
        ts = [r["t"] for r in sub]
        ax1.plot(ts, [r["cosine_sim"] for r in sub], marker="o", label=condition, color=color)
        ax2.plot(ts, [r["projection"] for r in sub], marker="o", label=condition, color=color)
    ax1.set_xlabel("turn (t)"); ax1.set_ylabel("cosine similarity"); ax1.set_title("Direction match to emotion vector")
    ax2.set_xlabel("turn (t)"); ax2.set_ylabel("projection (dot onto unit vector)"); ax2.set_title("Magnitude along emotion direction")
    ax1.legend(); ax2.legend()
    fig.suptitle(f"Pilot: '{PILOT_EMOTION}' vs neutral control, layer {layer}")
    fig.tight_layout()
    fig.savefig("pilot_decay_plot.png", dpi=150)
    print("Saved pilot_decay_plot.png")

    print("\nWhat to look for: the emotion condition should start higher than the "
          "neutral condition at t=0 and move toward it as t increases. If both lines "
          "are flat and overlapping throughout, the vector isn't showing up in "
          "generation even right after the story - a bigger problem than decay speed, "
          "worth revisiting extraction before scaling to C2.")


if __name__ == "__main__":
    main()
