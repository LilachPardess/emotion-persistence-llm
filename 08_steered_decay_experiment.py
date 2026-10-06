"""
Phase C3: steered persistence/decay on neutral text — no emotion stories.

For each neutral set in NEUTRAL_SETS, its 10 sentences are fed in one per
turn: sentence 1 -> reply, sentence 2 -> reply, ..., sentence 10 -> reply.
In each emotion condition, the emotion vector's unit direction, scaled to the
same fixed norm (STEER_STRENGTH) for every emotion, is ADDED into the residual
stream (CAA-style) while the model writes its first reply only (t=0); every
later reply is unsteered, asking whether the steered kick persists as more
neutral sentences arrive.

The unsteered condition runs the same sentences with no steering at all and
is the baseline. Measurements are always taken with hooks OFF (one clean
cached forward pass over the finished transcript), so cosine/projection
reflect the residual of the produced text, not the live steered activation.

Replies are sampled, so every conversation is repeated once per seed in SEEDS.
A steered run and the unsteered run with the same seed form a matched pair;
comparing unsteered runs across seeds gives the pure sampling-noise level.

6 neutral sets x (6 emotions + 1 unsteered) x 5 seeds = 210 conversations,
10 turns each. Each steered conversation is measured against its own emotion
vector; each unsteered conversation is measured against all 6.

Outputs, one pair per neutral set:
  - steered_neutral_sentences_<neutral_set>.csv
  - steered_neutral_sentences_<neutral_set>_plot.png  (steered vs unsteered, per emotion panel, mean over seeds)
Summarize across sets with 09_plot_decay_summary.py.

Usage:
    python 08_steered_decay_experiment.py
Requires stimuli.json, plus emotion_vectors_final.pt and config.json in emotion_extraction/.
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
    "non_emotional_natural_text_1",
    "non_emotional_natural_text_2",
    "neutral_baseline_stories_02",
    "neutral_baseline_stories_03",
    "neutral_baseline_stories_04",
    "neutral_baseline_stories_05",
]

MAX_NEW_TOKENS = 35
SEEDS = [0, 1, 2, 3, 4]
STEER_STRENGTH = 105.0        # L2 norm of the added vector, identical for every emotion (layer-14 residual norm is ~236)

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


def make_add_vec(vector, strength):
    """Steer by a fixed-length push along v's unit direction; ||v|| has no effect."""
    return (vector / vector.norm()) * strength


def run_conversation(model, sentences, max_new_tokens, seed, layer=None, add_vec=None):
    """One turn per sentence: append it, generate a reply. Steering only on the first reply."""
    torch.manual_seed(seed)
    tokens = None
    turns = []
    hook_name = f"blocks.{layer}.hook_resid_post" if layer is not None else None

    for t, sentence in enumerate(sentences):
        sentence_tokens = model.to_tokens(sentence, prepend_bos=(t == 0))
        tokens = sentence_tokens if tokens is None else torch.cat([tokens, sentence_tokens], dim=1)
        pre_len = tokens.shape[1]

        steer_this_turn = add_vec is not None and t == 0

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
            "sentence": sentence,
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


def plot_results(csv_path, plot_path, layer, strength, neutral_set):
    import pandas as pd
    df = pd.read_csv(csv_path)
    n_seeds = df["seed"].nunique()
    emotions = list(COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, emotion in zip(axes, emotions):
        steered = df[(df["condition"] == emotion) & (df["measured_against"] == emotion)]
        unsteered = df[(df["condition"] == "unsteered") & (df["measured_against"] == emotion)]
        steered = steered.groupby("t")["cosine_sim"].mean()
        unsteered = unsteered.groupby("t")["cosine_sim"].mean()
        ax.plot(steered.index, steered.values, marker="o", color=COLORS[emotion], label="steered at t=0")
        ax.plot(unsteered.index, unsteered.values, marker="s", color="#888888", label="unsteered")
        ax.set_title(emotion)
        ax.set_xlabel("turn (t) = sentence t+1")
        ax.axhline(0, color="black", linewidth=0.4)
        if emotion == emotions[0]:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine similarity")
    axes[3].set_ylabel("cosine similarity")
    fig.suptitle(
        f"Steered persistence/decay on {neutral_set} (layer {layer}, steered at t=0 only, added norm={strength})\n"
        f"colored = steered with that emotion; gray = no steering; both measured against the panel's vector; "
        f"mean over {n_seeds} seeds"
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
    print(f"Using device: {device}, layer: {layer}, strength={STEER_STRENGTH}")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    saved = torch.load(VECTORS_PATH, map_location=device, weights_only=False)
    emotion_vectors = saved["emotion_vectors"]
    emotions = list(emotion_vectors.keys())

    runs = []
    for neutral_set in NEUTRAL_SETS:
        for seed in SEEDS:
            for steer_emotion in emotions + [None]:
                runs.append({
                    "neutral_set": neutral_set,
                    "seed": seed,
                    "condition": steer_emotion or "unsteered",
                    "steer_emotion": steer_emotion,
                })

    total = len(runs)
    print(f"Running {total} conversations ({len(NEUTRAL_SETS)} neutral sets x "
          f"({len(emotions)} emotions + unsteered) x {len(SEEDS)} seeds)...\n")

    rows_by_set = {neutral_set: [] for neutral_set in NEUTRAL_SETS}
    t_start = time.time()
    for i, run in enumerate(runs):
        elapsed = time.time() - t_start
        print(f"[{i+1}/{total}] {run['neutral_set']} / seed {run['seed']} / {run['condition']} "
              f"- {elapsed:.0f}s elapsed")

        add_vec = None
        if run["steer_emotion"] is not None:
            vector = emotion_vectors[run["steer_emotion"]][layer].to(device)
            add_vec = make_add_vec(vector, STEER_STRENGTH)

        final_tokens, turns = run_conversation(
            model, stimuli[run["neutral_set"]], MAX_NEW_TOKENS, run["seed"],
            layer=layer, add_vec=add_vec,
        )
        turns = extract_turn_activations(model, final_tokens, turns, layer)

        target_emotions = [run["steer_emotion"]] if run["steer_emotion"] else emotions
        for emotion in target_emotions:
            vector = emotion_vectors[emotion][layer].to(device)
            for turn in turns:
                cosine_sim, projection = project(turn["mean_act"], vector)
                rows_by_set[run["neutral_set"]].append({
                    "neutral_set": run["neutral_set"],
                    "seed": run["seed"],
                    "condition": run["condition"],
                    "measured_against": emotion,
                    "t": turn["t"],
                    "steered_turn": int(turn["steered"]),
                    "steer_strength": STEER_STRENGTH if run["steer_emotion"] else 0.0,
                    "cosine_sim": cosine_sim,
                    "projection": projection,
                    "sentence": turn["sentence"],
                    "response_text": model.to_string(turn["response_tokens"]).strip(),
                })

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min.")

    for neutral_set in NEUTRAL_SETS:
        csv_path = f"steered_neutral_sentences_{neutral_set}.csv"
        plot_path = f"steered_neutral_sentences_{neutral_set}_plot.png"
        rows = rows_by_set[neutral_set]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "neutral_set", "seed", "condition", "measured_against", "t",
                "steered_turn", "steer_strength",
                "cosine_sim", "projection", "sentence", "response_text",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {csv_path}")
        plot_results(csv_path, plot_path, layer, STEER_STRENGTH, neutral_set)


if __name__ == "__main__":
    main()
