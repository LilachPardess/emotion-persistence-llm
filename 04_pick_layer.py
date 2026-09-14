"""
Phase B3/B4: pick the layer to use for the rest of the project.

At every layer, checks whether the 3 positive emotion vectors (happy, calm,
proud) cluster together, the 3 negative ones (sad, desperate, angry)
cluster together, and the two groups point in roughly opposite directions
(cross-valence cosine similarity should be clearly negative). This is the
"general emotion axis" your brainstorm was after - before trusting any
single emotion's vector, we want to see that positive vs negative separates
cleanly somewhere in the network.

Picks the layer that maximizes:
    gap = avg(within_positive_sim, within_negative_sim) - cross_valence_sim

Outputs:
  - emotion_vector_geometry.png   (layer sweep + heatmap at the chosen layer)
  - config.json                   ({"chosen_layer": N}) - C1/C2 read this
    automatically so you never have to retype a layer number by hand.

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

POSITIVE = ["happy", "calm", "proud"]
NEGATIVE = ["sad", "desperate", "angry"]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    saved = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
    emotion_vectors = saved["emotion_vectors"]  # dict[str] -> [n_layers, d_model]
    n_layers = saved["n_layers"]
    print(f"Loaded vectors for {list(emotion_vectors.keys())} across {n_layers} layers.")

    within_pos_list, within_neg_list, cross_list, gap_list = [], [], [], []
    for L in range(n_layers):
        wp = [cosine(emotion_vectors[a][L], emotion_vectors[b][L])
              for i, a in enumerate(POSITIVE) for b in POSITIVE[i + 1:]]
        wn = [cosine(emotion_vectors[a][L], emotion_vectors[b][L])
              for i, a in enumerate(NEGATIVE) for b in NEGATIVE[i + 1:]]
        cr = [cosine(emotion_vectors[a][L], emotion_vectors[b][L])
              for a in POSITIVE for b in NEGATIVE]
        wp_m, wn_m, cr_m = np.mean(wp), np.mean(wn), np.mean(cr)
        gap = (wp_m + wn_m) / 2 - cr_m
        within_pos_list.append(wp_m)
        within_neg_list.append(wn_m)
        cross_list.append(cr_m)
        gap_list.append(gap)
        print(f"  layer {L:2d}: within_pos={wp_m:+.3f}  within_neg={wn_m:+.3f}  "
              f"cross_valence={cr_m:+.3f}  gap={gap:+.3f}")

    best_layer = int(np.argmax(gap_list))
    print(f"\nChosen layer: {best_layer} (gap={gap_list[best_layer]:.3f}, "
          f"cross_valence_sim={cross_list[best_layer]:.3f})")
    if cross_list[best_layer] > -0.05:
        print("WARNING: even the best layer's cross-valence similarity isn't clearly "
              "negative. The positive/negative axis may be weak in this model at this "
              "scale - worth a manual look at the B2 playground before trusting Phase C.")

    # ---- Plot 1: layer sweep ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(n_layers), within_pos_list, marker="o", label="within-positive similarity")
    ax.plot(range(n_layers), within_neg_list, marker="o", label="within-negative similarity")
    ax.plot(range(n_layers), cross_list, marker="o", label="cross-valence similarity")
    ax.plot(range(n_layers), gap_list, marker="s", linestyle="--", color="gray", label="gap")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(best_layer, color="black", linestyle=":", alpha=0.5, label=f"chosen layer ({best_layer})")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Emotion vector geometry across layers")
    ax.legend()
    fig.tight_layout()
    fig.savefig("emotion_vector_geometry_layersweep.png", dpi=150)
    print("Saved emotion_vector_geometry_layersweep.png")

    # ---- Plot 2: full 6x6 heatmap at the chosen layer ----
    order = POSITIVE + NEGATIVE
    n = len(order)
    sim = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            sim[i, j] = cosine(emotion_vectors[a][best_layer], emotion_vectors[b][best_layer])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(order, rotation=45, ha="right")
    ax2.set_yticks(range(n)); ax2.set_yticklabels(order)
    ax2.axhline(2.5, color="black", linewidth=1); ax2.axvline(2.5, color="black", linewidth=1)
    ax2.set_title(f"Emotion vector cosine similarity (layer {best_layer})")
    fig2.colorbar(im, ax=ax2, label="cosine similarity")
    fig2.tight_layout()
    fig2.savefig("emotion_vector_geometry_heatmap.png", dpi=150)
    print("Saved emotion_vector_geometry_heatmap.png")

    with open(CONFIG_PATH, "w") as f:
        json.dump({"chosen_layer": best_layer}, f, indent=2)
    print(f"Wrote {CONFIG_PATH} with chosen_layer={best_layer} - Phase C scripts will read this automatically.")


if __name__ == "__main__":
    main()
