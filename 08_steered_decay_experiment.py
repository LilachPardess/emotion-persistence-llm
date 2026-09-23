"""
Phase C3: steered persistence/decay — same multi-turn protocol as 06, but
emotion conditions ADD the emotion vector into the residual stream during
generation (CAA-style), instead of relying on the story prompt alone.

Two steering schedules (set STEER_MODE):
  - "every_turn":  hook on for every model.generate call (t=0..10)
  - "t0_only":     hook on only for the first response; later turns unsteered
                   (asks whether a strong steered kick persists under fillers)

Neutral-control conversations never steer. Measurements are always taken
with hooks OFF (one clean cached forward pass over the finished transcript),
so cosine/projection reflect the residual of the produced text, not the
live steered activation.

Outputs:
  - steered_decay_results.csv
  - steered_decay_results_plot.png  (emotion vs neutral, per emotion panel)

Usage:
    python 08_steered_decay_experiment.py
Requires stimuli.json, emotion_vectors.pt, config.json.
"""
import csv
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
VECTORS_PATH = "emotion_vectors.pt"
CONFIG_PATH = "config.json"
OUTPUT_CSV = "steered_decay_t0_only_results.csv"
OUTPUT_PLOT = "steered_decay_t0_only_results_plot.png"

REPEAT_STORY_INDICES = [0, 5, 10]
MAX_NEW_TOKENS = 35
SEED = 0
STEER_STRENGTH = 5.0          # multiplier of native vector norm (same scale as 03 playground)
STEER_MODE = "t0_only"        # "every_turn" | "t0_only"

COLORS = {
    "happy": "#f4a259", "calm": "#8cb369", "proud": "#5b8e7d",
    "sad": "#4059ad", "desperate": "#6b2737", "angry": "#d1495b",
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_add_vec(vector, strength):
    """Steer by strength * ||v|| along the unit direction of v."""
    return (vector / vector.norm()) * vector.norm() * strength


def run_conversation(model, story_text, filler_texts, max_new_tokens, seed,
                     layer=None, add_vec=None, steer_mode="every_turn"):
    """Same token-space multi-turn loop as 06, with optional resid_post steering."""
    tokens = model.to_tokens(story_text)
    torch.manual_seed(seed)
    schedule = [None] + filler_texts
    turns = []
    hook_name = f"blocks.{layer}.hook_resid_post" if layer is not None else None

    for t, filler in enumerate(schedule):
        if filler is not None:
            filler_tokens = model.to_tokens(filler, prepend_bos=False)
            tokens = torch.cat([tokens, filler_tokens], dim=1)
        pre_len = tokens.shape[1]

        steer_this_turn = (
            add_vec is not None
            and (steer_mode == "every_turn" or (steer_mode == "t0_only" and t == 0))
        )

        with torch.no_grad():
            if steer_this_turn:
                def add_hook(resid, hook, _v=add_vec):
                    return resid + _v
                with model.hooks(fwd_hooks=[(hook_name, add_hook)]):
                    out = model.generate(
                        tokens, max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=0.8, stop_at_eos=False, verbose=False,
                    )
            else:
                out = model.generate(
                    tokens, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.8, stop_at_eos=False, verbose=False,
                )
        tokens = out
        turns.append({
            "t": t,
            "pre_len": pre_len,
            "post_len": tokens.shape[1],
            "steered": steer_this_turn,
        })
    return tokens, turns


def extract_turn_activations(model, final_tokens, turns, layer):
    """Clean (unhooked) forward pass; mean-pool each response span."""
    with torch.no_grad():
        _, cache = model.run_with_cache(
            final_tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
    acts = cache[f"blocks.{layer}.hook_resid_post"][0]
    for turn in turns:
        turn_acts = acts[turn["pre_len"]:turn["post_len"]]
        turn["mean_act"] = turn_acts.mean(dim=0)
        turn["response_tokens"] = final_tokens[0, turn["pre_len"]:turn["post_len"]]
    return turns


def project(mean_act, vector):
    unit = vector / vector.norm()
    cosine_sim = torch.nn.functional.cosine_similarity(
        mean_act.unsqueeze(0), vector.unsqueeze(0)
    ).item()
    projection = torch.dot(mean_act, unit).item()
    return cosine_sim, projection


def plot_results(csv_path, plot_path, layer, steer_mode, strength):
    import pandas as pd
    df = pd.read_csv(csv_path)
    emotions = ["happy", "calm", "proud", "sad", "desperate", "angry"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, emotion in zip(axes, emotions):
        emo = (df[(df["condition"] == emotion) & (df["measured_against"] == emotion)]
               .groupby("t")["cosine_sim"].agg(["mean", "sem"]))
        neu = (df[(df["condition"] == "neutral_control") & (df["measured_against"] == emotion)]
               .groupby("t")["cosine_sim"].agg(["mean", "sem"]))
        ax.plot(emo.index, emo["mean"], marker="o", color=COLORS[emotion], label="steered")
        ax.fill_between(emo.index, emo["mean"] - emo["sem"], emo["mean"] + emo["sem"],
                         color=COLORS[emotion], alpha=0.2)
        ax.plot(neu.index, neu["mean"], marker="s", color="#888888", label="neutral control")
        ax.fill_between(neu.index, neu["mean"] - neu["sem"], neu["mean"] + neu["sem"],
                         color="#888888", alpha=0.15)
        ax.set_title(emotion)
        ax.set_xlabel("turn (t)")
        ax.axhline(0, color="black", linewidth=0.4)
        if emotion == "happy":
            ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine similarity")
    axes[3].set_ylabel("cosine similarity")
    fig.suptitle(
        f"Steered persistence/decay (layer {layer}, {steer_mode}, α={strength})\n"
        f"colored = emotion story + steering; gray = neutral, no steering"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


def main():
    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    layer = config["chosen_layer"]

    device = get_device()
    print(f"Using device: {device}, layer: {layer}, "
          f"steer_mode={STEER_MODE}, strength={STEER_STRENGTH}")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    saved = torch.load(VECTORS_PATH, map_location=device, weights_only=False)
    emotion_vectors = saved["emotion_vectors"]
    emotions = list(emotion_vectors.keys())
    filler_texts = stimuli["neutral_filler_turns"]

    runs = []
    for emotion in emotions:
        for idx in REPEAT_STORY_INDICES:
            runs.append({
                "condition": emotion,
                "story_id": idx,
                "story_text": stimuli["emotions"][emotion][idx],
                "steer_emotion": emotion,
            })
    for idx in REPEAT_STORY_INDICES:
        runs.append({
            "condition": "neutral_control",
            "story_id": idx,
            "story_text": stimuli["neutral_baseline_stories"][idx],
            "steer_emotion": None,
        })

    total = len(runs)
    print(f"Running {total} conversations with steering on emotion conditions...\n")

    rows = []
    t_start = time.time()
    for i, run in enumerate(runs):
        elapsed = time.time() - t_start
        print(f"[{i+1}/{total}] {run['condition']} (story {run['story_id']}) - {elapsed:.0f}s elapsed")

        add_vec = None
        if run["steer_emotion"] is not None:
            vector = emotion_vectors[run["steer_emotion"]][layer].to(device)
            add_vec = make_add_vec(vector, STEER_STRENGTH)

        final_tokens, turns = run_conversation(
            model, run["story_text"], filler_texts, MAX_NEW_TOKENS, SEED,
            layer=layer, add_vec=add_vec, steer_mode=STEER_MODE,
        )
        turns = extract_turn_activations(model, final_tokens, turns, layer)

        target_emotions = emotions if run["condition"] == "neutral_control" else [run["condition"]]
        for emotion in target_emotions:
            vector = emotion_vectors[emotion][layer].to(device)
            for turn in turns:
                cosine_sim, projection = project(turn["mean_act"], vector)
                rows.append({
                    "condition": run["condition"],
                    "measured_against": emotion,
                    "story_id": run["story_id"],
                    "t": turn["t"],
                    "steered_turn": int(turn["steered"]),
                    "steer_mode": STEER_MODE,
                    "steer_strength": STEER_STRENGTH if run["steer_emotion"] else 0.0,
                    "cosine_sim": cosine_sim,
                    "projection": projection,
                    "response_text": model.to_string(turn["response_tokens"]).strip(),
                })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "measured_against", "story_id", "t",
            "steered_turn", "steer_mode", "steer_strength",
            "cosine_sim", "projection", "response_text",
        ])
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min. Saved {len(rows)} rows to {OUTPUT_CSV}")
    plot_results(OUTPUT_CSV, OUTPUT_PLOT, layer, STEER_MODE, STEER_STRENGTH)


if __name__ == "__main__":
    main()
