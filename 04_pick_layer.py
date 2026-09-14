"""
Phase B3/B4: pick the layer AND the contrast method to use for the rest of
the project.

Two ways to build a "specific emotion" vector from the activations B1 saved:
  - vs_neutral:  specific_emotion_mean - neutral_mean
                 (everything different about this emotion vs flat text -
                 bundles together "emotional at all" and "specifically this
                 emotion")
  - vs_general:  specific_emotion_mean - mean(all 6 emotions)
                 (removes the shared "emotional at all" component, leaving
                 what's distinctive about THIS emotion vs the others - the
                 "general emotion axis" idea from the original brainstorm)

Needs no model re-run - both are just arithmetic on the raw per-sentence
activations B1 already saved.

For each method, at every layer: checks whether the 3 positive vectors
(happy, calm, proud) cluster together, the 3 negative ones (sad, desperate,
angry) cluster together, and the two groups point in roughly opposite
directions (cross-valence cosine similarity should be clearly negative).
Picks whichever (method, layer) combination maximizes:
    gap = avg(within_positive_sim, within_negative_sim) - cross_valence_sim

Outputs:
  - emotion_vector_geometry_comparison.png   (both methods' layer sweeps)
  - emotion_vector_geometry_heatmap.png      (heatmap at the winning combo)
  - config.json          ({"chosen_layer": N, "chosen_method": "vs_neutral"|"vs_general"})
  - emotion_vectors_final.pt   (the winning vectors, in the same format
    emotion_vectors.pt uses, so C1/C2 can just point at this file instead)

Usage:
    python 04_pick_layer.py
Requires emotion_vectors.pt from 02_extract_emotion_vectors.py.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt

VECTORS_PATH = "emotion_vectors.pt"
CONFIG_PATH = "config.json"
FINAL_VECTORS_PATH = "emotion_vectors_final.pt"

POSITIVE = ["happy", "calm", "proud"]
NEGATIVE = ["sad", "desperate", "angry"]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def geometry_sweep(vectors_by_emotion, n_layers):
    """vectors_by_emotion: dict[emotion] -> [n_layers, d_model]. Returns
    (within_pos, within_neg, cross, gap) lists, one value per layer."""
    within_pos_list, within_neg_list, cross_list, gap_list = [], [], [], []
    for L in range(n_layers):
        wp = [cosine(vectors_by_emotion[a][L], vectors_by_emotion[b][L])
              for i, a in enumerate(POSITIVE) for b in POSITIVE[i + 1:]]
        wn = [cosine(vectors_by_emotion[a][L], vectors_by_emotion[b][L])
              for i, a in enumerate(NEGATIVE) for b in NEGATIVE[i + 1:]]
        cr = [cosine(vectors_by_emotion[a][L], vectors_by_emotion[b][L])
              for a in POSITIVE for b in NEGATIVE]
        wp_m, wn_m, cr_m = np.mean(wp), np.mean(wn), np.mean(cr)
        gap = (wp_m + wn_m) / 2 - cr_m
        within_pos_list.append(wp_m); within_neg_list.append(wn_m)
        cross_list.append(cr_m); gap_list.append(gap)
    return within_pos_list, within_neg_list, cross_list, gap_list


def main():
    saved = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
    n_layers = saved["n_layers"]
    neutral_mean = saved["neutral_mean"]                # [n_layers, d_model]
    emotion_raw_acts = saved["emotion_raw_acts"]          # dict[emotion] -> [15, n_layers, d_model]
    emotions = list(emotion_raw_acts.keys())
    print(f"Loaded raw activations for {emotions} across {n_layers} layers.")

    emotion_means = {e: acts.mean(dim=0) for e, acts in emotion_raw_acts.items()}  # [n_layers, d_model]
    general_mean = torch.stack([emotion_means[e] for e in emotions]).mean(dim=0)   # [n_layers, d_model]

    vectors_vs_neutral = {e: emotion_means[e] - neutral_mean for e in emotions}
    vectors_vs_general = {e: emotion_means[e] - general_mean for e in emotions}

    methods = {"vs_neutral": vectors_vs_neutral, "vs_general": vectors_vs_general}
    sweeps = {name: geometry_sweep(vecs, n_layers) for name, vecs in methods.items()}

    print("\nlayer | vs_neutral gap | vs_general gap")
    for L in range(n_layers):
        print(f"  {L:3d} |     {sweeps['vs_neutral'][3][L]:+.3f}     |    {sweeps['vs_general'][3][L]:+.3f}")

    # Pick the best (method, layer) combo by max gap
    best = {"method": None, "layer": None, "gap": -1e9}
    for name, (_, _, _, gap_list) in sweeps.items():
        L = int(np.argmax(gap_list))
        if gap_list[L] > best["gap"]:
            best = {"method": name, "layer": L, "gap": gap_list[L]}

    chosen_vectors = methods[best["method"]]
    cross_at_best = sweeps[best["method"]][2][best["layer"]]
    print(f"\nChosen: method={best['method']}, layer={best['layer']} "
          f"(gap={best['gap']:.3f}, cross_valence_sim={cross_at_best:.3f})")
    if cross_at_best > -0.05:
        print("WARNING: even the best (method, layer) combo isn't clearly negative on "
              "cross-valence similarity. Worth a manual look at the B2 playground with "
              "these specific vectors before trusting Phase C.")

    # ---- Plot 1: both methods' layer sweeps side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (name, (wp, wn, cr, gap)) in zip(axes, sweeps.items()):
        ax.plot(range(n_layers), wp, marker="o", label="within-positive")
        ax.plot(range(n_layers), wn, marker="o", label="within-negative")
        ax.plot(range(n_layers), cr, marker="o", label="cross-valence")
        ax.plot(range(n_layers), gap, marker="s", linestyle="--", color="gray", label="gap")
        ax.axhline(0, color="black", linewidth=0.5)
        if name == best["method"]:
            ax.axvline(best["layer"], color="black", linestyle=":", alpha=0.6, label=f"chosen ({best['layer']})")
        ax.set_title(name)
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine similarity")
    fig.suptitle("Emotion vector geometry: vs_neutral vs. vs_general contrast")
    fig.tight_layout()
    fig.savefig("emotion_vector_geometry_comparison.png", dpi=150)
    print("Saved emotion_vector_geometry_comparison.png")

    # ---- Plot 2: heatmap at the winning (method, layer) ----
    order = POSITIVE + NEGATIVE
    n = len(order)
    sim = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            sim[i, j] = cosine(chosen_vectors[a][best["layer"]], chosen_vectors[b][best["layer"]])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(order, rotation=45, ha="right")
    ax2.set_yticks(range(n)); ax2.set_yticklabels(order)
    ax2.axhline(2.5, color="black", linewidth=1); ax2.axvline(2.5, color="black", linewidth=1)
    ax2.set_title(f"Emotion vector cosine similarity\n(method={best['method']}, layer={best['layer']})")
    fig2.colorbar(im, ax=ax2, label="cosine similarity")
    fig2.tight_layout()
    fig2.savefig("emotion_vector_geometry_heatmap.png", dpi=150)
    print("Saved emotion_vector_geometry_heatmap.png")

    with open(CONFIG_PATH, "w") as f:
        json.dump({"chosen_layer": best["layer"], "chosen_method": best["method"]}, f, indent=2)
    print(f"Wrote {CONFIG_PATH}")

    torch.save({
        "model_name": saved["model_name"],
        "n_layers": n_layers,
        "d_model": saved["d_model"],
        "emotion_vectors": chosen_vectors,   # full [n_layers, d_model] per emotion, winning method
        "method": best["method"],
    }, FINAL_VECTORS_PATH)
    print(f"Saved {FINAL_VECTORS_PATH} - Phase C scripts should load vectors from this file now, "
          f"not the original emotion_vectors.pt.")


if __name__ == "__main__":
    main()
