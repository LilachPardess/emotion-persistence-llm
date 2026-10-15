"""
Reading test: do the 6 emotion vectors pick out the right emotion in text?

For every sentence, the vector with the highest cosine similarity is the
predicted emotion. Score: % of sentences where it is the labeled emotion
(chance = 1/6). Predictions and setup: emotion_extraction/VALIDATION.md.

1. Datasets
   - training:  the 120 Alex sentences the vectors were built from
                (stimuli.json["emotions"], 20 per emotion). Sanity check
                only, so it should be close to 100%. Activations are read from
                emotion_vectors.pt ("emotion_raw_acts"), so no model run.
   - held-out:  30 Alex sentences generated the same way but not used to
                build the vectors (stimuli.json["emotions_heldout"], 5 per
                emotion).
   - GoEmotions: human-labeled Reddit comments
                (goemotions_validation/goemotions_100_per_emotion.csv),
                pre-filtered to single-label comments that raters agreed on,
                at least 4 words, up to 100 per label. Only the 6 labels that
                match the vectors are used.

2. Activations
   GPT-2 Medium, blocks.{layer}.hook_resid_post at the layer from config.json
   (14), at the last token - the same as 02_extract_emotion_vectors.py.

3. Scoring (same for every dataset)
   Subtract the dataset's own average activation from each sentence, then
   take the cosine with each vector. The vectors are used as saved: 02
   already subtracts the average over all emotions.

4. Checks
   - Accuracy: % correct per emotion and overall, per dataset.
   - Confusion matrix per dataset (rows: true emotion, columns: predicted),
     plus each emotion's share of all predictions, to spot one emotion that
     wins whatever the label.
   - GoEmotions only, as before: Test 2 (point-biserial r between
     label == e and cos_e across comments) and the label x vector mean-cosine
     heatmap.
   - Vector-to-vector cosine heatmap: exploratory, no prediction.

Outputs:
  - emotion_extraction/results/raw/reading_accuracy.csv
      (% correct per emotion x dataset, with n and chance)
  - emotion_extraction/results/raw/reading_predictions.csv
      (one row per sentence: dataset, label, text, 6 cosines, prediction)
  - emotion_extraction/results/plots/reading_confusion.png
      (one confusion matrix per dataset)
  - emotion_extraction/results/plots/goemotions_validation_last_token_heatmap.png
      (GoEmotions label x vector mean cosine)
  - emotion_extraction/results/plots/vector_similarity_heatmap.png
      (exploratory)
Test 2 is printed only.

Usage (from the repo root):
    python emotion_extraction/11_validate_vectors_on_text.py
Requires stimuli.json, emotion_extraction/goemotions_validation/
goemotions_100_per_emotion.csv, and emotion_vectors.pt,
emotion_vectors_final.pt and config.json in emotion_extraction/results/raw/
(from 02_extract_emotion_vectors.py). Takes ~1-2 min: the model loads once and
encodes the held-out sentences and GoEmotions comments.
"""
import json
import os
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"
STIMULI_PATH = "stimuli.json"
RAW_DIR = "emotion_extraction/results/raw"
PLOTS_DIR = "emotion_extraction/results/plots"
CONFIG_PATH = f"{RAW_DIR}/config.json"
RAW_VECTORS_PATH = f"{RAW_DIR}/emotion_vectors.pt"
FINAL_VECTORS_PATH = f"{RAW_DIR}/emotion_vectors_final.pt"
GOEMOTIONS_PATH = "emotion_extraction/goemotions_validation/goemotions_100_per_emotion.csv"
ACCURACY_PATH = f"{RAW_DIR}/reading_accuracy.csv"
CONFUSION_PATH = f"{PLOTS_DIR}/reading_confusion.png"
# Sequential blue ramp (light = near 0%, dark = near 100%) for the confusion matrices.
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
# Training sentences re-encoded to check encoding matches 02: one per emotion.
# A mismatch (model, layer, tokenization) would show on every sentence, so a few suffice.
N_ENCODING_CHECK_SENTENCES = 6


def load_chosen_layer():
    with open(CONFIG_PATH) as f:
        return json.load(f)["chosen_layer"]


def load_emotion_vectors(chosen_layer):
    """{emotion: [d_model] vector at `chosen_layer`}, on the CPU, in the saved order."""
    saved = torch.load(FINAL_VECTORS_PATH, map_location="cpu", weights_only=False)
    return {e: v[chosen_layer] for e, v in saved["emotion_vectors"].items()}


def print_load_summary(dataset, source, sentences, emotions):
    """One line: how many (text, emotion) pairs were loaded, from where, per emotion."""
    counts = Counter(e for _, e in sentences)
    print(f"Loaded {len(sentences)} {dataset} from {source}: "
          + ", ".join(f"{e}={counts[e]}" for e in emotions))


def load_training_sentences(emotions, chosen_layer):
    """[(text, emotion)] for the 120 Alex sentences the vectors were built from,
    and their saved [n, d_model] activations at `chosen_layer` (same order)."""
    with open(STIMULI_PATH) as f:
        training = json.load(f)["emotions"]
    raw_acts = torch.load(RAW_VECTORS_PATH, map_location="cpu",
                          weights_only=False)["emotion_raw_acts"]
    for e in emotions:
        if len(training[e]) != len(raw_acts[e]):
            raise ValueError(
                f"{e}: {len(training[e])} sentences in {STIMULI_PATH} but "
                f"{len(raw_acts[e])} saved activations in {RAW_VECTORS_PATH}; "
                f"rerun 02_extract_emotion_vectors.py"
            )
    sentences = [(text, e) for e in emotions for text in training[e]]
    activations = torch.cat([raw_acts[e][:, chosen_layer] for e in emotions])
    print_load_summary("training sentences", f"{STIMULI_PATH} (emotions)", sentences, emotions)
    return sentences, activations


def load_heldout_sentences(emotions):
    """[(text, emotion)] for the held-out Alex sentences."""
    with open(STIMULI_PATH) as f:
        heldout = json.load(f)["emotions_heldout"]
    sentences = [(text, e) for e in emotions for text in heldout.get(e, [])]
    print_load_summary("held-out sentences", f"{STIMULI_PATH} (emotions_heldout)",
                       sentences, emotions)
    return sentences


def clean_goemotions_comments(comments, emotions):
    """Trim spaces, keep only non-empty comments labeled with one of our
    emotions, drop repeats. Returns (cleaned comments, number of repeats dropped)."""
    comments["sentence"] = comments["sentence"].astype(str).str.strip()
    comments = comments[(comments["sentence"] != "") & comments["emotion"].isin(emotions)]
    n_with_our_labels = len(comments)
    comments = comments.drop_duplicates(subset="sentence")
    return comments, n_with_our_labels - len(comments)


def load_goemotions_sentences(emotions):
    """[(text, emotion)] for GoEmotions comments whose label is one of `emotions`."""
    comments, n_duplicates = clean_goemotions_comments(pd.read_csv(GOEMOTIONS_PATH), emotions)
    sentences = list(zip(comments["sentence"], comments["emotion"]))
    print_load_summary("GoEmotions comments",
                       f"{GOEMOTIONS_PATH} ({n_duplicates} duplicates dropped)",
                       sentences, emotions)
    return sentences


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    device = get_device()
    print(f"Loading {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()
    return model


def encode_last_token(model, sentences, chosen_layer):
    """[n, d_model] residual stream at `chosen_layer`, last token, on the CPU -
    read the same way as 02_extract_emotion_vectors.py."""
    hook_name = f"blocks.{chosen_layer}.hook_resid_post"
    activations = []
    for text, _ in sentences:
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(text),
                                            names_filter=lambda n: n == hook_name)
        activations.append(cache[hook_name][0, -1].cpu())
    return torch.stack(activations)



def validate_encoding_matches_saved_activations(model, training_sentences, training_activations, chosen_layer):
    """Re-encode a few training sentences (spread over all emotions) and compare
    with the activations 02 saved, so held-out and GoEmotions are read exactly
    like the vectors."""
    step = len(training_sentences) // N_ENCODING_CHECK_SENTENCES
    checked_indices = list(range(0, len(training_sentences), step))[:N_ENCODING_CHECK_SENTENCES]
    recomputed_activations = encode_last_token(
        model, [training_sentences[i] for i in checked_indices], chosen_layer
    )
    similarities = torch.nn.functional.cosine_similarity(
        recomputed_activations, training_activations[checked_indices], dim=1
    )
    lowest_similarity = similarities.min().item()
    print(f"Encoding check vs saved activations: min cosine over {N_ENCODING_CHECK_SENTENCES} "
          f"sentences = {lowest_similarity:.6f}")
    if lowest_similarity < 0.999:
        raise ValueError("Activations differ from the ones 02 saved; "
                         "check the model, layer or tokenization.")


def score_dataset(sentences, activations, emotion_vectors):
    """One row per sentence: text, label, cosine with each emotion vector,
    predicted emotion (highest cosine) and whether it matches the label.
    Activations are centered on the dataset's own average first."""
    emotions = list(emotion_vectors)
    centered_activations = activations - activations.mean(dim=0)
    stacked_emotion_vectors = torch.stack([emotion_vectors[e] for e in emotions])  # [6, d_model]
    cosines = torch.nn.functional.cosine_similarity(
        centered_activations.unsqueeze(1), stacked_emotion_vectors.unsqueeze(0), dim=2
    )  # [n_sentences, 6]
    scores = pd.DataFrame(cosines.numpy(), columns=[f"cos_{e}" for e in emotions])
    scores.insert(0, "text", [text for text, _ in sentences])
    scores.insert(1, "label", [e for _, e in sentences])
    scores["predicted"] = [emotions[i] for i in cosines.argmax(dim=1).tolist()]
    scores["correct"] = scores["predicted"] == scores["label"]
    return scores


def accuracy_table(scores, emotions):
    """One row per dataset x emotion, plus an "overall" row per dataset:
    n sentences, n correct, % correct and chance (1 / number of emotions)."""
    rows = []
    for dataset, dataset_scores in scores.items():
        groups = [(e, dataset_scores[dataset_scores["label"] == e]) for e in emotions]
        groups.append(("overall", dataset_scores))
        for emotion, group in groups:
            rows.append({
                "dataset": dataset,
                "emotion": emotion,
                "n": len(group),
                "n_correct": int(group["correct"].sum()),
                "accuracy": group["correct"].mean(),
                "chance": 1 / len(emotions),
            })
    return pd.DataFrame(rows)


def print_accuracy_table(accuracy, emotions):
    """Emotions as rows, datasets as columns: '% correct (n)'."""
    cells = accuracy.assign(
        cell=[f"{a:4.0%} (n={n})" for a, n in zip(accuracy["accuracy"], accuracy["n"])]
    )
    table = cells.pivot(index="emotion", columns="dataset", values="cell")
    table = table.loc[emotions + ["overall"], list(dict.fromkeys(accuracy["dataset"]))]
    print(f"\n% correct (highest-cosine vector = label); chance = "
          f"{accuracy['chance'].iloc[0]:.0%}")
    print(table.to_string())


def report_accuracy(scores, emotions):
    """Build the accuracy table (output 5.1), print it and save it to ACCURACY_PATH."""
    accuracy = accuracy_table(scores, emotions)
    print_accuracy_table(accuracy, emotions)
    os.makedirs(RAW_DIR, exist_ok=True)
    accuracy.to_csv(ACCURACY_PATH, index=False, float_format="%.4f")
    print(f"\nSaved {ACCURACY_PATH}")
    return accuracy


def confusion_matrix(dataset_scores, emotions):
    """Rows: true emotion, columns: predicted emotion; each row sums to 1."""
    counts = pd.crosstab(dataset_scores["label"], dataset_scores["predicted"])
    counts = counts.reindex(index=emotions, columns=emotions, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def prediction_shares(scores, emotions):
    """Share of all sentences predicted as each emotion, per dataset (columns).
    Balanced datasets make 1 / number of emotions the expected share."""
    return pd.DataFrame({
        dataset: dataset_scores["predicted"].value_counts(normalize=True)
                                            .reindex(emotions, fill_value=0)
        for dataset, dataset_scores in scores.items()
    })


def print_confusion(scores, emotions):
    for dataset, dataset_scores in scores.items():
        matrix = confusion_matrix(dataset_scores, emotions)
        print(f"\nConfusion matrix, {dataset} (rows: true, columns: predicted, % of row)")
        print((matrix * 100).round().astype(int).to_string())
    shares = prediction_shares(scores, emotions)
    print(f"\nShare of all predictions per emotion (expected {1 / len(emotions):.0%} each)")
    print(shares.apply(lambda col: col.map("{:.0%}".format)).to_string())


def plot_confusion_matrices(scores, emotions, chosen_layer, path):
    """One panel per dataset, same 0-100% color scale; diagonal = correct."""
    fig, axes = plt.subplots(1, len(scores), figsize=(5.2 * len(scores), 5.2),
                             constrained_layout=True)
    for ax, (dataset, dataset_scores) in zip(axes, scores.items()):
        matrix = confusion_matrix(dataset_scores, emotions).to_numpy()
        image = ax.imshow(matrix, cmap=CONFUSION_CMAP, vmin=0, vmax=1)
        for i in range(len(emotions)):
            for j in range(len(emotions)):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 0.55 else "#1f1f1f")
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#1f1f1f", linewidth=1.5))
        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels(emotions, rotation=45, ha="right")
        ax.set_yticks(range(len(emotions)))
        ax.set_yticklabels(emotions)
        ax.set_xlabel("predicted (highest-cosine vector)")
        ax.set_ylabel("true emotion")
        n = len(dataset_scores)
        ax.set_title(f"{dataset} (n={n}): {dataset_scores['correct'].mean():.0%} correct")
    fig.colorbar(image, ax=axes, shrink=0.8, label="% of the row's sentences",
                 format=lambda x, _: f"{x:.0%}")
    fig.suptitle(f"Which vector wins, by true emotion (layer {chosen_layer}, "
                 f"chance = {1 / len(emotions):.0%}; boxes = correct)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")


def report_confusion(scores, emotions, chosen_layer):
    """Print the confusion matrices and prediction shares, and save the figure."""
    print_confusion(scores, emotions)
    plot_confusion_matrices(scores, emotions, chosen_layer, CONFUSION_PATH)


def main():
    model = load_model()

    chosen_layer = load_chosen_layer()
    emotion_vectors = load_emotion_vectors(chosen_layer)
    emotions = list(emotion_vectors)
    training_sentences, training_activations = load_training_sentences(emotions, chosen_layer)
    heldout_sentences = load_heldout_sentences(emotions)
    goemotions_sentences = load_goemotions_sentences(emotions)

    validate_encoding_matches_saved_activations(model, training_sentences, training_activations, chosen_layer)
    print("Encoding held-out sentences and GoEmotions comments...")
    
    datasets = {
        "training": (training_sentences, training_activations),
        "held-out": (heldout_sentences,
                     encode_last_token(model, heldout_sentences, chosen_layer)),
        "GoEmotions": (goemotions_sentences,
                       encode_last_token(model, goemotions_sentences, chosen_layer)),
    }
    scores = {name: score_dataset(sentences, activations, emotion_vectors)
              for name, (sentences, activations) in datasets.items()}

    report_accuracy(scores, emotions)
    report_confusion(scores, emotions, chosen_layer)


if __name__ == "__main__":
    main()
