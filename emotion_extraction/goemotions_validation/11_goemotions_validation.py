"""
Out-of-distribution check on GoEmotions (human-annotated Reddit comments):
do the emotion vectors line up with real web text they were not built from?

1. Dataset (goemotions_100_per_emotion.csv, next to this script)
   Pre-filtered GoEmotions: single-label comments with high rater agreement
   and enough words for context, up to 100 per label. Columns: sentence,
   emotion. Only the labels in TARGET_LABELS are used, and only their
   matching vectors are scored ("desperate" has no GoEmotions counterpart).

2. Activation extraction
   One forward pass per comment; read blocks.{layer}.hook_resid_post at the
   layer from config.json, at the last token (matches how 02 builds the
   vectors).

3. Baseline normalization
   Subtract the global mean activation over all selected comments from each
   comment, leaving only its semantic/emotional displacement from a typical
   comment.
   With CENTER_VECTORS, also subtract the mean of the mapped emotion vectors
   from each vector, removing the direction they all share so each keeps
   only what distinguishes it from the others.

4. Correlation metric
   Cosine similarity between each normalized comment vector and each mapped
   emotion vector. Evaluated with:
     Test 1 (within comment): for comments whose label maps to vector e,
       matching_gap = cos_e - mean(cos of the other vectors), and whether
       vector e is the top-scoring vector.
     Test 2 (across comments): for each vector e, point-biserial r between
       (label == TARGET_LABELS[e]) and cos_e over all selected comments.
     Heatmap: mean cosine per label x vector; the diagonal is the expected match.

Outputs (written to emotion_extraction/goemotions_validation/, named by
settings, so runs with each setting sit side by side):
  - goemotions_validation_last_token[_centered_vectors]_results.csv   (one row per comment)
  - goemotions_validation_last_token[_centered_vectors]_plot.png      (Test 1 + Test 2)
  - goemotions_validation_last_token[_centered_vectors]_heatmap.png   (label x vector mean cosine)

Usage (from the repo root):
    python emotion_extraction/goemotions_validation/11_goemotions_validation.py
Requires goemotions_100_per_emotion.csv (in this folder) plus
emotion_vectors_final.pt and config.json (in emotion_extraction/).
"""
import csv
import json
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
VALIDATION_DIR = "emotion_extraction/goemotions_validation"
DATASET_PATH = f"{VALIDATION_DIR}/goemotions_100_per_emotion.csv"
FINAL_VECTORS_PATH = "emotion_extraction/emotion_vectors_final.pt"
CONFIG_PATH = "emotion_extraction/config.json"
CENTER_VECTORS = True
RUN_NAME = "last_token_centered_vectors" if CENTER_VECTORS else "last_token"
OUTPUT_PREFIX = f"{VALIDATION_DIR}/goemotions_validation_{RUN_NAME}"

TARGET_LABELS = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "proud": "pride",
    "calm": "relief",
}

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


def load_comments():
    df = pd.read_csv(DATASET_PATH).rename(columns={"sentence": "text", "emotion": "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].drop_duplicates(subset="text")
    df = df[df["label"].isin(TARGET_LABELS.values())].reset_index(drop=True)
    print(f"Loaded {len(df)} comments from {DATASET_PATH} "
          f"(labels: {', '.join(TARGET_LABELS.values())})")
    return df


def get_last_token_resid(model, text, layer, device):
    tokens = model.to_tokens(text).to(device)
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
    return cache[hook_name][0, -1].cpu()


def correlation_stats(labels, scores):
    """Point-biserial r (= Pearson r with a 0/1 label), Fisher-z 95% CI, and a
    two-sided p-value (normal approximation)."""
    labels, scores = np.asarray(labels, float), np.asarray(scores, float)
    n = len(labels)
    r = float(np.corrcoef(labels, scores)[0, 1])
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    return {
        "r": r,
        "ci_low": math.tanh(z - 1.96 * se),
        "ci_high": math.tanh(z + 1.96 * se),
        "p": math.erfc(abs(t) / math.sqrt(2)),
    }


def evaluate(comments, cos, emotions):
    """cos: DataFrame [n_comments, n_vectors] of normalized cosines."""
    test1 = {}
    for e in emotions:
        mask = (comments["label"] == TARGET_LABELS[e]).to_numpy()
        sub = cos[mask]
        others = sub[[o for o in emotions if o != e]]
        gaps = sub[e] - others.mean(axis=1)
        is_max = sub.idxmax(axis=1) == e
        test1[e] = {"n": int(mask.sum()), "mean_gap": float(gaps.mean()),
                    "frac_gap_pos": float((gaps > 0).mean()),
                    "frac_is_max": float(is_max.mean())}

    test2 = {}
    for e in emotions:
        labels = (comments["label"] == TARGET_LABELS[e]).astype(int).to_numpy()
        s = correlation_stats(labels, cos[e].to_numpy())
        s["mean_match"] = float(cos.loc[labels == 1, e].mean())
        s["mean_other"] = float(cos.loc[labels == 0, e].mean())
        test2[e] = s
    return test1, test2


def plot_tests(test1, test2, emotions, layer, method, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [COLORS.get(e, "#888888") for e in emotions]
    xlabels = [f"{e}\n({TARGET_LABELS[e]})\nn={test1[e]['n']}" for e in emotions]

    ax = axes[0]
    gaps = [test1[e]["mean_gap"] for e in emotions]
    bars = ax.bar(xlabels, gaps, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{g:+.3f}\ntop={test1[e]['frac_is_max']:.0%}"
                               for g, e in zip(gaps, emotions)], fontsize=8, padding=2)
    ax.margins(y=0.15)
    ax.set_ylabel("mean matching gap (cos_e - mean cos_other)")
    ax.set_title(f"Test 1: matching vs other vectors (within comment)\n"
                 f"top = vector e scores highest (chance {1 / len(emotions):.0%})")

    ax = axes[1]
    rs = [test2[e]["r"] for e in emotions]
    err = [[test2[e]["r"] - test2[e]["ci_low"] for e in emotions],
           [test2[e]["ci_high"] - test2[e]["r"] for e in emotions]]
    bars = ax.bar(xlabels, rs, yerr=err, capsize=4, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{r:+.3f}" for r in rs], fontsize=9, padding=3)
    ax.margins(y=0.15)
    ax.set_ylabel("point-biserial r  (label == target vs cos_e)")
    ax.set_title("Test 2: same vector across comments\n(label e vs the other labels, 95% CI)")

    fig.suptitle(f"GoEmotions validation (method={method}, layer={layer}, "
                 f"{RUN_NAME}, comments globally mean-centered)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def plot_heatmap(comments, cos, emotions, layer, path):
    row_labels = [TARGET_LABELS[e] for e in emotions]
    table = cos.groupby(comments["label"].to_numpy()).mean().loc[row_labels, emotions]
    counts = comments["label"].value_counts()
    values = table.to_numpy()
    lim = float(np.abs(values).max())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"{lab} (n={counts[lab]})" for lab in row_labels])
    for i in range(len(row_labels)):
        for j in range(len(emotions)):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=1.5))
    ax.set_xlabel("emotion vector")
    ax.set_ylabel("GoEmotions label")
    ax.set_title(f"Mean cosine (centered comment vs vector)\n"
                 f"layer={layer}, {RUN_NAME}; boxes = expected match")
    fig.colorbar(im, ax=ax, label="mean cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    layer = config["chosen_layer"]
    method = config.get("chosen_method", "unknown")

    vectors = torch.load(FINAL_VECTORS_PATH, map_location="cpu", weights_only=False)["emotion_vectors"]
    emotions = [e for e in vectors if e in TARGET_LABELS]
    vector_matrix = torch.stack([vectors[e][layer] for e in emotions])  # [n_vectors, d_model]
    if CENTER_VECTORS:
        vector_matrix = vector_matrix - vector_matrix.mean(dim=0)

    comments = load_comments()
    counts = comments["label"].value_counts()
    print(f"method={method}, layer={layer}, center_vectors={CENTER_VECTORS}")
    for e in emotions:
        print(f"  {e:10s} <- {TARGET_LABELS[e]:8s} n={counts.get(TARGET_LABELS[e], 0):4d}")

    device = get_device()
    print(f"\nUsing device: {device}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    X = torch.stack([get_last_token_resid(model, text, layer, device)
                     for text in comments["text"]])  # [n_comments, d_model]
    X = X - X.mean(dim=0)
    cos_t = torch.nn.functional.cosine_similarity(
        X.unsqueeze(1), vector_matrix.unsqueeze(0), dim=2
    )  # [n_comments, n_vectors]
    cos = pd.DataFrame(cos_t.numpy(), columns=emotions)

    test1, test2 = evaluate(comments, cos, emotions)

    print(f"\nTest 1: within comment, matching vector vs the other {len(emotions) - 1}")
    print(f"{'vector':10s} {'label':9s} {'n':>4s} {'mean_gap':>9s} {'gap>0':>6s} {'top':>6s}")
    for e in emotions:
        t = test1[e]
        print(f"{e:10s} {TARGET_LABELS[e]:9s} {t['n']:4d} {t['mean_gap']:+9.4f} "
              f"{t['frac_gap_pos']:6.0%} {t['frac_is_max']:6.0%}")
    print(f"(chance for 'top' = {1 / len(emotions):.0%})")

    print("\nTest 2: across comments, point-biserial r (label == target) vs cos_e")
    print(f"{'vector':10s} {'label':9s} {'match':>8s} {'other':>8s} {'r':>7s}  "
          f"{'95% CI':>17s}  {'p':>9s}")
    for e in emotions:
        s = test2[e]
        print(f"{e:10s} {TARGET_LABELS[e]:9s} {s['mean_match']:+8.4f} {s['mean_other']:+8.4f} "
              f"{s['r']:+7.3f}  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  {s['p']:9.2g}")

    out = comments[["label", "text"]].copy()
    for e in emotions:
        out[f"cos_{e}"] = cos[e].to_numpy()
    out.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\nSaved {OUTPUT_PREFIX}_results.csv")

    plot_tests(test1, test2, emotions, layer, method, f"{OUTPUT_PREFIX}_plot.png")
    plot_heatmap(comments, cos, emotions, layer, f"{OUTPUT_PREFIX}_heatmap.png")


if __name__ == "__main__":
    main()
