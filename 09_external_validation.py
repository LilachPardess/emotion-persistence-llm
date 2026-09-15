"""
External (out-of-sample) correlation check: do the emotion vectors line up
with real-world tweets they were not built from?

Runs 40 tweets (external_validation.json) through the model, mean-pools
residual-stream activations at the chosen layer, and scores each sentence
against all 6 emotion vectors (cosine + projection).

Two correlation-style tests (not argmax classification):

  1. Matching vs other (within sentence)
     For a tweet labeled emotion e, is its score on vector e higher than its
     scores on the other vectors? Reports matching_gap =
     cos_e - mean(cos_other) and the fraction of tweets with gap > 0.

  2. Same vector, different labels (across sentences)
     For each covered emotion e, do tweets labeled e score higher on vector e
     than tweets labeled something else? Reports mean match vs mismatch
     cosine and the point-biserial correlation between (label == e) and cos_e.

Only 4 of 6 emotions have external ground truth (happy, sad, angry,
desperate). Scores are still computed against all 6 vectors.

Outputs:
  - external_validation_results.csv
  - external_validation_correlation.png  (two-panel summary)

Usage:
    python 09_external_validation.py
Requires: emotion_vectors_final.pt + config.json (from 04_pick_layer.py)
and external_validation.json, all in the same folder.
"""
import csv
import json
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
FINAL_VECTORS_PATH = "emotion_vectors_final.pt"
CONFIG_PATH = "config.json"
EXTERNAL_PATH = "external_validation.json"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_mean_resid_at_layer(model, text, layer, device):
    tokens = model.to_tokens(text).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post"
        )
    return cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0).cpu()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def point_biserial(binary, continuous):
    """Pearson r between a 0/1 label and a continuous score."""
    n = len(binary)
    if n < 2:
        return float("nan")
    n1 = sum(binary)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    m1 = mean([c for b, c in zip(binary, continuous) if b])
    m0 = mean([c for b, c in zip(binary, continuous) if not b])
    m = mean(continuous)
    sd = math.sqrt(sum((c - m) ** 2 for c in continuous) / n)
    if sd == 0:
        return float("nan")
    return ((m1 - m0) / sd) * math.sqrt((n1 * n0) / (n * n))


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    layer = config["chosen_layer"]
    method = config.get("chosen_method", "unknown")

    final_saved = torch.load(FINAL_VECTORS_PATH, map_location="cpu", weights_only=False)
    vectors = final_saved["emotion_vectors"]
    all_emotions = list(vectors.keys())

    with open(EXTERNAL_PATH) as f:
        external = json.load(f)
    covered_emotions = list(external["sentences"].keys())

    print(f"method={method}, layer={layer}")
    print(f"External ground truth covers: {covered_emotions}")
    print(f"Scoring against all vectors: {all_emotions}\n")

    device = get_device()
    print(f"Using device: {device}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    rows = []
    for true_emotion in covered_emotions:
        for sent in external["sentences"][true_emotion]:
            act = get_mean_resid_at_layer(model, sent, layer, device)
            cos_scores, proj_scores = {}, {}
            for e in all_emotions:
                vec = vectors[e][layer]
                cos_scores[e] = cosine(act, vec)
                proj_scores[e] = torch.dot(act, vec).item()

            others = [cos_scores[e] for e in all_emotions if e != true_emotion]
            matching_gap = cos_scores[true_emotion] - mean(others)
            match_is_max = int(cos_scores[true_emotion] == max(cos_scores.values()))

            row = {
                "true_emotion": true_emotion,
                "sentence": sent,
                "matching_cos": cos_scores[true_emotion],
                "other_cos_mean": mean(others),
                "matching_gap": matching_gap,
                "match_is_max": match_is_max,
            }
            for e in all_emotions:
                row[f"cos_{e}"] = cos_scores[e]
                row[f"proj_{e}"] = proj_scores[e]
            rows.append(row)

    # --- Test 1: matching vs other (within sentence) ---
    print("\n=== Test 1: matching vector vs other vectors (within sentence) ===")
    print("matching_gap = cos(true) - mean(cos(other vectors))")
    gaps_by_emotion = {e: [] for e in covered_emotions}
    max_by_emotion = {e: [] for e in covered_emotions}
    for r in rows:
        gaps_by_emotion[r["true_emotion"]].append(r["matching_gap"])
        max_by_emotion[r["true_emotion"]].append(r["match_is_max"])

    all_gaps = [r["matching_gap"] for r in rows]
    all_max = [r["match_is_max"] for r in rows]
    print(f"Overall: mean gap={mean(all_gaps):+.4f}, "
          f"gap>0: {sum(g > 0 for g in all_gaps)}/{len(all_gaps)} = "
          f"{mean([g > 0 for g in all_gaps]):.0%}, "
          f"match is max: {sum(all_max)}/{len(all_max)} = {mean(all_max):.0%}")
    for e in covered_emotions:
        gaps = gaps_by_emotion[e]
        print(f"  {e:10s}: mean gap={mean(gaps):+.4f}, "
              f"gap>0={sum(g > 0 for g in gaps)}/{len(gaps)}, "
              f"match max={sum(max_by_emotion[e])}/{len(max_by_emotion[e])}")

    # --- Test 2: same vector, different labels (across sentences) ---
    print("\n=== Test 2: same vector across labels (match tweets vs other tweets) ===")
    print("point-biserial r between (label == e) and cos_e")
    test2 = {}
    for e in covered_emotions:
        labels = [int(r["true_emotion"] == e) for r in rows]
        scores = [r[f"cos_{e}"] for r in rows]
        match_scores = [s for lab, s in zip(labels, scores) if lab]
        other_scores = [s for lab, s in zip(labels, scores) if not lab]
        r_pb = point_biserial(labels, scores)
        test2[e] = {
            "mean_match": mean(match_scores),
            "mean_other": mean(other_scores),
            "delta": mean(match_scores) - mean(other_scores),
            "r": r_pb,
        }
        print(f"  {e:10s}: match={test2[e]['mean_match']:+.4f}, "
              f"other={test2[e]['mean_other']:+.4f}, "
              f"delta={test2[e]['delta']:+.4f}, r={test2[e]['r']:+.3f}")

    pos_deltas = sum(1 for e in covered_emotions if test2[e]["delta"] > 0)
    pos_rs = sum(1 for e in covered_emotions if test2[e]["r"] > 0)
    print(f"\nSummary: {pos_deltas}/{len(covered_emotions)} emotions have match > other mean; "
          f"{pos_rs}/{len(covered_emotions)} have r > 0.")
    if mean(all_gaps) <= 0 and pos_deltas <= len(covered_emotions) / 2:
        print("WARNING: little evidence that vectors correlate with these tweets. "
              "They may be fitting writing style more than emotion.")
    else:
        print("Some positive alignment with tweet labels — check per-emotion bars "
              "(real tweets are noisier than hand-written stimuli).")

    with open("external_validation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved external_validation_results.csv")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: mean matching gap by true emotion
    ax = axes[0]
    gap_means = [mean(gaps_by_emotion[e]) for e in covered_emotions]
    colors = ["#5b8e7d" if g > 0 else "#a44a3f" for g in gap_means]
    bars = ax.bar(covered_emotions, gap_means, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mean matching gap (cos)")
    ax.set_title("Test 1: matching vs other vectors\n(within sentence)")
    ax.bar_label(bars, labels=[f"{g:+.3f}" for g in gap_means], fontsize=9)

    # Panel 2: mean cos on vector e for match vs other tweets
    ax = axes[1]
    x = range(len(covered_emotions))
    width = 0.35
    match_means = [test2[e]["mean_match"] for e in covered_emotions]
    other_means = [test2[e]["mean_other"] for e in covered_emotions]
    b1 = ax.bar([i - width / 2 for i in x], match_means, width, label="tweets labeled e", color="#5b8e7d")
    b2 = ax.bar([i + width / 2 for i in x], other_means, width, label="tweets labeled ≠ e", color="#888888")
    ax.set_xticks(list(x))
    ax.set_xticklabels(covered_emotions)
    ax.set_ylabel(f"mean cosine onto vector e")
    ax.set_title("Test 2: same vector, different labels\n(across sentences)")
    ax.legend(fontsize=8)
    for i, e in enumerate(covered_emotions):
        ax.annotate(f"r={test2[e]['r']:+.2f}", (i, max(match_means[i], other_means[i])),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    fig.suptitle(f"External tweet–vector correlation (method={method}, layer={layer})", y=1.02)
    fig.tight_layout()
    fig.savefig("external_validation_correlation.png", dpi=150, bbox_inches="tight")
    print("Saved external_validation_correlation.png")


if __name__ == "__main__":
    main()
