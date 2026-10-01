"""
Phase C2: scale the C1 pilot up to all 6 emotions + a neutral control, each
with 3 repeat stories (spread across different topics), and save every
turn's measurement to CSV.

The experiment is repeated once per neutral set in NEUTRAL_SETS, each using
3 control stories from that set. Every conversation reseeds, so the emotion
conversations are identical across the three and are generated only once.

27 conversations total (6 emotions x 3 repeats + 3 neutral sets x 3 control
stories), 11 measurement turns each. The neutral control conversations
are each measured against ALL 6 emotion vectors (cheap - it's just extra
cache passes, not extra generation), so every emotion gets a matched
"nothing was induced" reference curve built from the exact same fillers/seed.

Same run_conversation / activation-extraction logic as 05_pilot_decay.py,
already verified there and further optimized here (cache each conversation's
activations once, project onto every needed vector from that single cache).

Outputs, one pair per neutral set:
  - decay_results_<neutral_set>.csv   (long format: one row per turn - condition,
                          measured_against, story_id, t, cosine_sim,
                          projection, response_text)
  - decay_results_<neutral_set>_plot.png  (emotion vs neutral, per emotion panel)

Usage:
    python 06_full_decay_experiment.py
Requires stimuli.json, plus emotion_vectors_final.pt and config.json in emotion_extraction/.
Takes roughly 10-20 min on CPU - it prints progress as it goes.
"""
import csv
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
VECTORS_PATH = "emotion_extraction/emotion_vectors_final.pt"
CONFIG_PATH = "emotion_extraction/config.json"
NEUTRAL_SETS = [
    "non_emotional_natural_text",
    "neutral_baseline_stories_02",
    "neutral_baseline_stories_03",
]

REPEAT_STORY_INDICES = [0, 5, 10]  # spread across different topics for diversity
MAX_NEW_TOKENS = 35
SEED = 0

COLORS = {
    "joy": "#f4a259", "admiration": "#8cb369", "optimism": "#5b8e7d",
    "sadness": "#4059ad", "anger": "#d1495b", "fear": "#6b2737",
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_conversation(model, story_text, filler_texts, max_new_tokens, seed):
    """Builds one full conversation turn by turn, entirely in token space."""
    tokens = model.to_tokens(story_text)
    torch.manual_seed(seed)
    schedule = [None] + filler_texts
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


def extract_turn_activations(model, final_tokens, turns, layer):
    """One cached forward pass over the whole conversation; adds mean-pooled
    response activations and response tokens to each turn dict."""
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


def plot_results(csv_path, plot_path, layer, neutral_set):
    import pandas as pd
    df = pd.read_csv(csv_path)
    emotions = list(COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, emotion in zip(axes, emotions):
        emo = (df[(df["condition"] == emotion) & (df["measured_against"] == emotion)]
               .groupby("t")["cosine_sim"].agg(["mean", "sem"]))
        neu = (df[(df["condition"] == "neutral_control") & (df["measured_against"] == emotion)]
               .groupby("t")["cosine_sim"].agg(["mean", "sem"]))
        ax.plot(emo.index, emo["mean"], marker="o", color=COLORS[emotion], label="emotion story")
        ax.fill_between(emo.index, emo["mean"] - emo["sem"], emo["mean"] + emo["sem"],
                         color=COLORS[emotion], alpha=0.2)
        ax.plot(neu.index, neu["mean"], marker="s", color="#888888", label="neutral control")
        ax.fill_between(neu.index, neu["mean"] - neu["sem"], neu["mean"] + neu["sem"],
                         color="#888888", alpha=0.15)
        ax.set_title(emotion)
        ax.set_xlabel("turn (t)")
        ax.axhline(0, color="black", linewidth=0.4)
        if emotion == emotions[0]:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine similarity")
    axes[3].set_ylabel("cosine similarity")
    fig.suptitle(
        f"Emotion persistence/decay (layer {layer}, neutral control: {neutral_set})\n"
        f"colored = emotion story; gray = neutral story, measured against the same vector"
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
    print(f"Using device: {device}, layer: {layer}")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    saved = torch.load(VECTORS_PATH, map_location=device, weights_only=False)
    emotion_vectors = saved["emotion_vectors"]
    emotions = list(emotion_vectors.keys())
    filler_texts = stimuli["neutral_filler_turns"]

    runs = []
    for emotion in emotions:
        for idx in REPEAT_STORY_INDICES:
            runs.append({"condition": emotion, "story_id": idx, "neutral_set": None,
                         "story_text": stimuli["emotions"][emotion][idx]})
    for neutral_set in NEUTRAL_SETS:
        for idx in REPEAT_STORY_INDICES:
            runs.append({"condition": "neutral_control", "story_id": idx,
                         "neutral_set": neutral_set,
                         "story_text": stimuli[neutral_set][idx]})

    total = len(runs)
    print(f"Running {total} conversations ({len(emotions)} emotions x "
          f"{len(REPEAT_STORY_INDICES)} + {len(NEUTRAL_SETS)} neutral sets x "
          f"{len(REPEAT_STORY_INDICES)} neutral control), "
          f"{len(filler_texts) + 1} turns each...\n")

    emotion_rows = []
    control_rows = {neutral_set: [] for neutral_set in NEUTRAL_SETS}
    t_start = time.time()
    for i, run in enumerate(runs):
        elapsed = time.time() - t_start
        label = run["neutral_set"] or run["condition"]
        print(f"[{i+1}/{total}] {label} (story {run['story_id']}) - {elapsed:.0f}s elapsed")

        final_tokens, turns = run_conversation(model, run["story_text"], filler_texts, MAX_NEW_TOKENS, SEED)
        turns = extract_turn_activations(model, final_tokens, turns, layer)

        # Which emotion vector(s) to measure this conversation against
        target_emotions = emotions if run["condition"] == "neutral_control" else [run["condition"]]
        rows = control_rows[run["neutral_set"]] if run["neutral_set"] else emotion_rows

        for emotion in target_emotions:
            vector = emotion_vectors[emotion][layer].to(device)
            for turn in turns:
                cosine_sim, projection = project(turn["mean_act"], vector)
                rows.append({
                    "condition": run["condition"],
                    "measured_against": emotion,
                    "story_id": run["story_id"],
                    "t": turn["t"],
                    "cosine_sim": cosine_sim,
                    "projection": projection,
                    "response_text": model.to_string(turn["response_tokens"]).strip(),
                })

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min.")

    for neutral_set in NEUTRAL_SETS:
        csv_path = f"decay_results_{neutral_set}.csv"
        plot_path = f"decay_results_{neutral_set}_plot.png"
        rows = emotion_rows + control_rows[neutral_set]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "condition", "measured_against", "story_id", "t",
                "cosine_sim", "projection", "response_text",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {csv_path}")
        plot_results(csv_path, plot_path, layer, neutral_set)


if __name__ == "__main__":
    main()
