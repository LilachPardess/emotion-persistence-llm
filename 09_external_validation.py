"""
External (out-of-sample) validation: does the emotion vector generalize
beyond the 90 hand-written stimuli sentences it was built from?

Runs 40 real-world tweets (external_validation.json, built from a labeled
Twitter emotion dataset - completely separate from stimuli.json, written
by different people about different topics) through the model, computes
each sentence's mean-pooled residual-stream activation at the chosen
layer, and projects it onto all 6 emotion vectors - the exact same
classification rule used for the in-sample accuracy check
(04_pick_layer.py / 08_per_emotion_accuracy.py).

Only 4 of the 6 emotions have external ground truth here (happy, sad,
angry, desperate - the source dataset has no calm/proud tweets), so
accuracy is scored only over those 4 true labels. Predictions still range
over all 6 vectors, so a tweet CAN be predicted as calm or proud - that's
not a bug, there's just no ground truth to check it against.

Why this matters: the in-sample ~50% accuracy could mean either (a) the
vectors capture real emotion structure, or (b) they capture this
project's own narrow writing style (topic, sentence length, register)
which the classifier is fitting rather than emotion itself. This tweet set
was NOT written by us and NOT hand-picked for how clearly it expresses the
emotion (filtered only for objective quality - see external_validation.json's
"note" field) - so if external accuracy holds up meaningfully above chance,
that is real evidence for (a). If it collapses to chance, that's evidence
for (b), and the in-sample number was likely optimistic.

Outputs:
  - external_validation_results.csv   (one row per tweet: true_emotion,
    predicted_emotion, correct, cosine sim + projection onto all 6 vectors)
  - external_validation_accuracy.png  (bar chart: in-sample accuracy vs.
    external accuracy, overall and broken down by the 4 covered emotions)

Usage:
    python 09_external_validation.py
Requires: emotion_vectors_final.pt + config.json (from 04_pick_layer.py),
emotion_vectors.pt (for the in-sample comparison number), and
external_validation.json, all in the same folder.
"""
import csv
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
FINAL_VECTORS_PATH = "emotion_vectors_final.pt"
RAW_PATH = "emotion_vectors.pt"
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


def in_sample_accuracy(vectors, layer):
    """Recompute the in-sample number (same rule, same vectors) for the
    comparison bar, straight from the raw per-sentence activations."""
    raw_saved = torch.load(RAW_PATH, map_location="cpu", weights_only=False)
    emotion_raw_acts = raw_saved["emotion_raw_acts"]
    emotions = list(emotion_raw_acts.keys())
    correct, total = 0, 0
    for true_emotion in emotions:
        acts = emotion_raw_acts[true_emotion][:, layer, :]
        for i in range(acts.shape[0]):
            scores = {e: torch.dot(acts[i], vectors[e][layer]).item() for e in emotions}
            pred = max(scores, key=scores.get)
            total += 1
            correct += int(pred == true_emotion)
    return correct / total


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
    covered_emotions = list(external["sentences"].keys())  # happy, sad, angry, desperate

    print(f"method={method}, layer={layer}")
    print(f"External ground truth covers: {covered_emotions}")
    print(f"Predicting across all vectors: {all_emotions}\n")

    device = get_device()
    print(f"Using device: {device}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    rows = []
    confusion = {t: {p: 0 for p in all_emotions} for t in covered_emotions}
    correct_total, total = 0, 0
    per_emotion_correct = {e: 0 for e in covered_emotions}
    per_emotion_total = {e: 0 for e in covered_emotions}

    for true_emotion in covered_emotions:
        for sent in external["sentences"][true_emotion]:
            act = get_mean_resid_at_layer(model, sent, layer, device)
            scores, cos_scores = {}, {}
            for e in all_emotions:
                vec = vectors[e][layer]
                scores[e] = torch.dot(act, vec).item()
                cos_scores[e] = cosine(act, vec)
            pred = max(scores, key=scores.get)
            is_correct = int(pred == true_emotion)

            confusion[true_emotion][pred] += 1
            correct_total += is_correct
            total += 1
            per_emotion_correct[true_emotion] += is_correct
            per_emotion_total[true_emotion] += 1

            row = {"true_emotion": true_emotion, "predicted_emotion": pred,
                   "correct": is_correct, "sentence": sent}
            for e in all_emotions:
                row[f"cos_{e}"] = cos_scores[e]
                row[f"proj_{e}"] = scores[e]
            rows.append(row)

    ext_accuracy = correct_total / total
    chance = 1 / len(all_emotions)
    print(f"\nExternal (out-of-sample) accuracy: {correct_total}/{total} = {ext_accuracy:.1%} "
          f"(chance={chance:.1%}, scored against {len(covered_emotions)} of {len(all_emotions)} emotions)")

    print("\nPer-emotion external accuracy:")
    for e in covered_emotions:
        acc = per_emotion_correct[e] / per_emotion_total[e]
        print(f"  {e:10s}: {per_emotion_correct[e]:2d}/{per_emotion_total[e]} = {acc:.0%}")

    print("\nConfusion (true -> predicted counts, only non-zero shown):")
    for t in covered_emotions:
        row_str = ", ".join(f"{p}={c}" for p, c in confusion[t].items() if c > 0)
        print(f"  {t:10s}: {row_str}")

    in_sample_acc = in_sample_accuracy(vectors, layer)
    print(f"\nIn-sample accuracy   (90 hand-written stimuli, same method/layer): {in_sample_acc:.1%}")
    print(f"External accuracy    (40 real-world tweets, 4 emotions):           {ext_accuracy:.1%}")
    if ext_accuracy < chance * 1.5:
        print("\nWARNING: external accuracy is close to chance. The vectors may be picking up "
              "this project's own writing style/topics rather than emotion itself - treat the "
              "in-sample number with caution and mention this limitation in the write-up.")
    else:
        print("\nExternal accuracy clears chance by a meaningful margin: some real generalization "
              "beyond the hand-written stimuli. Real-world tweets are noisier (sarcasm, mixed "
              "signals, crowd-labeling errors) so some drop from the in-sample number is expected "
              "regardless of vector quality.")

    with open("external_validation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved external_validation_results.csv")

    labels = ["overall\n(in-sample)", "overall\n(external)"] + [f"{e}\n(external)" for e in covered_emotions]
    values = [in_sample_acc, ext_accuracy] + [per_emotion_correct[e] / per_emotion_total[e] for e in covered_emotions]
    bar_colors = ["#888888", "#5b8e7d"] + ["#5b8e7d"] * len(covered_emotions)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=bar_colors)
    ax.axhline(chance, color="black", linestyle=":", label=f"chance ({chance:.1%})")
    ax.set_ylabel("classification accuracy")
    ax.set_title(f"In-sample vs. external validation accuracy\n(method={method}, layer={layer})")
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in values])
    ax.legend()
    fig.tight_layout()
    fig.savefig("external_validation_accuracy.png", dpi=150)
    print("Saved external_validation_accuracy.png")


if __name__ == "__main__":
    main()
