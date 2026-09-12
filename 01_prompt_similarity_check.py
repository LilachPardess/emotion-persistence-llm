"""
Sanity check #2: how does the model represent similar vs. different prompts?

We take 8 short, emotion-neutral prompts across 4 topics (2 prompts per
topic - so same-topic pairs are "similar", cross-topic pairs are
"different"), pull the residual-stream activation for each, center them
(subtract the mean across prompts - removes the shared residual-stream
direction that otherwise makes every pair look ~0.99 similar), and plot:

  1. A pairwise cosine-similarity heatmap (prompts grouped by topic).
     Same-topic pairs should look brighter than cross-topic pairs -
     that's the basic assumption the whole emotion-vector method leans on.
  2. A layer sweep: at every layer, average within-topic similarity vs.
     average cross-topic similarity. This shows which layers separate
     "same idea" from "different idea" most clearly - useful later for
     picking which layer to build the actual emotion vectors from.

Run this after 00_sanity_check.py. Uses the same model, so gpt2-medium
will already be cached locally - no re-download.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"  # keep in sync with whatever 00_sanity_check.py used

# 4 topics x 2 prompts each. All emotion-neutral on purpose - this check is
# about semantic similarity, not emotion yet.
PROMPTS = {
    "weather": [
        "The sky turned dark as clouds rolled over the mountains.",
        "Rain began to fall steadily across the valley.",
    ],
    "sports": [
        "The team practiced drills every morning before school.",
        "She scored the winning goal in the final minute of the match.",
    ],
    "cooking": [
        "He chopped the onions finely before adding them to the pan.",
        "The recipe called for two cups of flour and a pinch of salt.",
    ],
    "technology": [
        "The new phone features a faster processor and a better camera.",
        "Software updates are rolled out automatically every month.",
    ],
}


def get_all_layer_resids(model, text, n_layers):
    """One forward pass, mean-pooled residual stream at every layer."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("hook_resid_post")
        )
    return [cache[f"blocks.{L}.hook_resid_post"][0].mean(dim=0) for L in range(n_layers)]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


if __name__ == "__main__":
    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    n_layers = model.cfg.n_layers

    # 1. Flatten prompts and topics
    flat_prompts = []
    topics = []
    for topic, sentences in PROMPTS.items():
        for sentence in sentences:
            flat_prompts.append(sentence)
            topics.append(topic)

    num_prompts = len(flat_prompts)

    # 2. Extract residual stream for every prompt at every layer
    print("Extracting residuals...")
    # all_resids shape: [num_prompts, n_layers, d_model]
    all_resids = []
    for text in flat_prompts:
        resids = get_all_layer_resids(model, text, n_layers)
        all_resids.append(torch.stack(resids))
    all_resids = torch.stack(all_resids)

    # 3. Center: subtract mean across prompts to remove shared direction
    mean_resids = all_resids.mean(dim=0, keepdim=True)
    centered_resids = all_resids - mean_resids

    # 4. Heatmap of pairwise cosine similarity at a mid/late layer
    layer_to_plot = 12
    sim_matrix = np.zeros((num_prompts, num_prompts))
    for i in range(num_prompts):
        for j in range(num_prompts):
            sim_matrix[i, j] = cosine(
                centered_resids[i, layer_to_plot],
                centered_resids[j, layer_to_plot],
            )

    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(num_prompts), topics, rotation=45)
    plt.yticks(range(num_prompts), topics)
    plt.title(f"Pairwise Prompt Similarity (Layer {layer_to_plot})")
    plt.tight_layout()
    plt.savefig("prompt_similarity_heatmap.png", dpi=150)
    print("Saved prompt_similarity_heatmap.png")
    plt.close()

    # 5. Layer sweep: within-topic vs cross-topic similarity
    within_topic_sims = []
    cross_topic_sims = []

    for L in range(n_layers):
        within_sim = []
        cross_sim = []
        for i in range(num_prompts):
            for j in range(i + 1, num_prompts):
                sim = cosine(centered_resids[i, L], centered_resids[j, L])
                if topics[i] == topics[j]:
                    within_sim.append(sim)
                else:
                    cross_sim.append(sim)
        within_topic_sims.append(np.mean(within_sim))
        cross_topic_sims.append(np.mean(cross_sim))

    plt.figure(figsize=(10, 5))
    plt.plot(range(n_layers), within_topic_sims, label="Within-topic (Similar)", marker="o")
    plt.plot(range(n_layers), cross_topic_sims, label="Cross-topic (Different)", marker="x")
    plt.xlabel("Layer")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Semantic Separation Across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("prompt_similarity_layer_sweep.png", dpi=150)
    print("Saved prompt_similarity_layer_sweep.png")
    plt.close()

    gaps = np.array(within_topic_sims) - np.array(cross_topic_sims)
    best_layer = int(np.argmax(gaps))
    print(f"Layer {layer_to_plot}: within={within_topic_sims[layer_to_plot]:.3f}, "
          f"cross={cross_topic_sims[layer_to_plot]:.3f}, "
          f"gap={gaps[layer_to_plot]:.3f}")
    print(f"Layer with largest within/cross gap: {best_layer} (gap={gaps[best_layer]:.3f})")
