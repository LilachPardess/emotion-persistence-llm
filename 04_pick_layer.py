"""
Phase B3/B4: pick the layer for the vs_general emotion vectors that 02 builds.

02 now defines emotion vectors as:
    emotion_mean - mean(all emotion means)   (= vs_general)

This script still plots vs_neutral alongside as a reference, but the chosen
layer, heatmap, config.json, and emotion_vectors_final.pt all use vs_general
so they stay aligned with 02.

Layer choice is NOT "max geometry gap over all layers" (that wrongly prefers
layer 0 when every gap is negative). Instead:
  1. Restrict candidates to mid/late layers (skip early residual stream).
  2. Score each candidate with a 6-way linear probe: project each sentence
     activation onto the 6 emotion vectors, predict argmax.
  3. Pick the layer with highest probe accuracy; break ties with geometry gap.

Geometry gap is still plotted as a diagnostic:
    gap = avg(within_positive_sim, within_negative_sim) - cross_valence_sim

Outputs:
  - emotion_vector_geometry_comparison.png   (both methods' layer sweeps)
  - emotion_vector_geometry_heatmap.png      (vs_general at the chosen layer)
  - config.json          ({"chosen_layer", "chosen_method", "probe_accuracy"})
  - emotion_vectors_final.pt   (vs_general vectors, same format as
    emotion_vectors.pt, so C1/C2 can just point at this file instead)

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
CHOSEN_METHOD = "vs_general"  # must match 02_extract_emotion_vectors.py

POSITIVE = ["happy", "calm", "proud"]
NEGATIVE = ["sad", "desperate", "angry"]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def candidate_layers(n_layers):
    """Mid/late residual stream only. For gpt2-medium (24 layers): 6..22."""
    lo = max(1, n_layers // 4)
    hi = max(lo + 1, n_layers - 1)  # exclusive end -> last included is n_layers-2
    return list(range(lo, hi))


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


def probe_accuracy(emotion_raw_acts, vectors, layer, emotions):
    """6-way linear probe: for each sentence, predict argmax projection onto
    the emotion vectors at this layer. Chance = 1/n_emotions."""
    correct, total = 0, 0
    for true_emotion in emotions:
        acts = emotion_raw_acts[true_emotion][:, layer, :]  # [n_sents, d_model]
        for i in range(acts.shape[0]):
            sentence_act = acts[i]
            scores = {e: torch.dot(sentence_act, vectors[e][layer]).item() for e in emotions}
            pred = max(scores, key=scores.get)
            correct += int(pred == true_emotion)
            total += 1
    return correct / total


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

    # Lock to vs_general; pick mid/late layer by probe accuracy (gap tiebreak).
    chosen_vectors = methods[CHOSEN_METHOD]
    gap_list = sweeps[CHOSEN_METHOD][3]
    layers = candidate_layers(n_layers)
    chance = 1 / len(emotions)

    print(f"\nCandidate layers for {CHOSEN_METHOD}: {layers[0]}..{layers[-1]} "
          f"(skipped early layers; chance probe accuracy = {chance:.1%})")
    print(f"{'layer':>6s} {'accuracy':>10s} {'gap':>8s}")
    probe_acc = {}
    for L in layers:
        acc = probe_accuracy(emotion_raw_acts, chosen_vectors, L, emotions)
        probe_acc[L] = acc
        print(f"{L:6d} {acc:9.1%} {gap_list[L]:+8.3f}")

    best_layer = max(layers, key=lambda L: (probe_acc[L], gap_list[L]))
    best = {
        "method": CHOSEN_METHOD,
        "layer": best_layer,
        "gap": gap_list[best_layer],
        "probe_accuracy": probe_acc[best_layer],
    }

    cross_at_best = sweeps[best["method"]][2][best["layer"]]
    print(f"\nChosen: method={best['method']}, layer={best['layer']} "
          f"(probe_acc={best['probe_accuracy']:.1%}, gap={best['gap']:.3f}, "
          f"cross_valence_sim={cross_at_best:.3f})")
    if best["probe_accuracy"] <= chance + 0.05:
        print("WARNING: probe accuracy is barely above chance. Emotion vectors may not "
              "carry usable emotion-specific signal at this layer.")
    if cross_at_best > -0.05:
        print("WARNING: cross-valence similarity isn't clearly negative. Worth a manual "
              "look at the B2 playground before trusting Phase C.")

    # ---- Plot 1: both methods' layer sweeps + probe accuracy for vs_general ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (name, (wp, wn, cr, gap)) in zip(axes, sweeps.items()):
        ax.plot(range(n_layers), wp, marker="o", label="within-positive")
        ax.plot(range(n_layers), wn, marker="o", label="within-negative")
        ax.plot(range(n_layers), cr, marker="o", label="cross-valence")
        ax.plot(range(n_layers), gap, marker="s", linestyle="--", color="gray", label="gap")
        ax.axhline(0, color="black", linewidth=0.5)
        if name == best["method"]:
            ax.axvline(best["layer"], color="black", linestyle=":", alpha=0.6,
                       label=f"chosen ({best['layer']})")
            ax.axvspan(layers[0], layers[-1], color="green", alpha=0.06, label="candidate band")
        ax.set_title(name)
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine similarity")
    fig.suptitle("Emotion vector geometry: vs_neutral vs. vs_general contrast\n"
                 f"layer chosen by mid/late probe accuracy ({best['probe_accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig("emotion_vector_geometry_comparison.png", dpi=150)
    print("Saved emotion_vector_geometry_comparison.png")

    # ---- Plot 2: heatmap at the chosen vs_general layer ----
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
    ax2.set_title(f"Emotion vector cosine similarity\n"
                  f"(method={best['method']}, layer={best['layer']}, "
                  f"probe={best['probe_accuracy']:.1%})")
    fig2.colorbar(im, ax=ax2, label="cosine similarity")
    fig2.tight_layout()
    fig2.savefig("emotion_vector_geometry_heatmap.png", dpi=150)
    print("Saved emotion_vector_geometry_heatmap.png")

    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "chosen_layer": best["layer"],
            "chosen_method": best["method"],
            "probe_accuracy": best["probe_accuracy"],
        }, f, indent=2)
    print(f"Wrote {CONFIG_PATH}")

    torch.save({
        "model_name": saved["model_name"],
        "n_layers": n_layers,
        "d_model": saved["d_model"],
        "emotion_vectors": chosen_vectors,   # full [n_layers, d_model] per emotion, vs_general
        "method": best["method"],
        "chosen_layer": best["layer"],
        "probe_accuracy": best["probe_accuracy"],
    }, FINAL_VECTORS_PATH)
    print(f"Saved {FINAL_VECTORS_PATH} - Phase C scripts should load vectors from this file now, "
          f"not the original emotion_vectors.pt.")


if __name__ == "__main__":
    main()
