"""
Phase C2: scale the C1 pilot up to all 6 emotions + a neutral control, each
with 3 repeat stories (spread across different topics), and save every
turn's measurement to CSV.

21 conversations total (6 emotions x 3 repeats + 3 neutral control stories),
6 measurement turns each = 126 generations. The neutral control conversations
are each measured against ALL 6 emotion vectors (cheap - it's just extra
cache passes, not extra generation), so every emotion gets a matched
"nothing was induced" reference curve built from the exact same fillers/seed.

Same run_conversation / activation-extraction logic as 05_pilot_decay.py,
already verified there and further optimized here (cache each conversation's
activations once, project onto every needed vector from that single cache).

Outputs:
  - decay_results.csv   (long format: one row per turn - condition,
                          measured_against, story_id, t, cosine_sim,
                          projection, response_text)

Usage:
    python 06_full_decay_experiment.py
Requires stimuli.json, emotion_vectors.pt, config.json.
Takes roughly 10-20 min on CPU - it prints progress as it goes.
"""
import csv
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
VECTORS_PATH = "emotion_vectors.pt"
CONFIG_PATH = "config.json"
OUTPUT_CSV = "decay_results.csv"

REPEAT_STORY_INDICES = [0, 5, 10]  # spread across different topics for diversity
MAX_NEW_TOKENS = 35
SEED = 0


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
            runs.append({"condition": emotion, "story_id": idx,
                         "story_text": stimuli["emotions"][emotion][idx]})
    for idx in REPEAT_STORY_INDICES:
        runs.append({"condition": "neutral_control", "story_id": idx,
                     "story_text": stimuli["neutral_baseline_stories"][idx]})

    total = len(runs)
    print(f"Running {total} conversations ({len(emotions)} emotions x "
          f"{len(REPEAT_STORY_INDICES)} + {len(REPEAT_STORY_INDICES)} neutral control), "
          f"6 turns each...\n")

    rows = []
    t_start = time.time()
    for i, run in enumerate(runs):
        elapsed = time.time() - t_start
        print(f"[{i+1}/{total}] {run['condition']} (story {run['story_id']}) - {elapsed:.0f}s elapsed")

        final_tokens, turns = run_conversation(model, run["story_text"], filler_texts, MAX_NEW_TOKENS, SEED)
        turns = extract_turn_activations(model, final_tokens, turns, layer)

        # Which emotion vector(s) to measure this conversation against
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
                    "cosine_sim": cosine_sim,
                    "projection": projection,
                    "response_text": model.to_string(turn["response_tokens"]).strip(),
                })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "measured_against", "story_id", "t",
            "cosine_sim", "projection", "response_text",
        ])
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min. Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
