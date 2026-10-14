"""
Reading test: do the 6 emotion vectors pick out the right emotion in text?

For every sentence, the vector with the highest cosine similarity is the
predicted emotion. Score: % of sentences where it is the labeled emotion
(chance = 1/6). Predictions and setup: emotion_extraction/VALIDATION.md.

1. Datasets
   - ours:      the 120 Alex sentences the vectors were built from
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
    python emotion_extraction/11_reading_validation.py
Requires stimuli.json, emotion_extraction/goemotions_validation/
goemotions_100_per_emotion.csv, and emotion_vectors.pt,
emotion_vectors_final.pt and config.json in emotion_extraction/results/raw/
(from 02_extract_emotion_vectors.py). Takes ~1-2 min: the model loads once and
encodes the held-out sentences and GoEmotions comments.
"""
