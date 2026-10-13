# Validating the emotion vectors

## Step 1: Write down what I expect, before looking again

**The question:** Do my 6 emotion vectors (joy, admiration, optimism, sadness, anger, fear) really represent these emotions?

### Predictions

**Reading texts**

- **Our dataset (sanity check).** For the sentences I used to build the vectors, the correct emotion will get the highest score for almost all of them (close to 100%). This only shows the code works, because the vectors were made from these same sentences.
- **GoEmotions.** For GoEmotions sentences, the correct emotion will get the highest score for about 70% of them. Random guessing gives 17% (1 in 6), so I expect clearly more.
- **Held-out sentences.** For new sentences not used to build the vectors, the correct emotion will get the highest score for about 95%, a bit lower than for our dataset.

**Steering**

- **Finishing a sentence.** With steering, the text will show the added emotion more than without steering. A random vector with the same strength will not create a clear emotion. To check fairly, I will mix the outputs, hide which emotion was added, and guess. I expect to guess right for about 50% of them (random guessing: 17%).
- **Rating question.** I will steer with each emotion, and each time ask about all 6 emotions with this prompt: `On a scale from 1 to 100, Alex's level of {emotion} was`. I expect the rating for the added emotion to go up more than the ratings for the other 5.

**Other**

- **Cosine similarity heatmap:** no prediction yet. I'm not sure how the emotions relate to each other (for example, whether joy is closer to optimism or to fear), so for now I won't use this test to check the vectors.

### What would surprise me

- A random vector with the same strength changes the text as much as an emotion vector.
- The vectors work on our dataset but are no better than 17% on GoEmotions.
- One emotion gets the highest score on almost every GoEmotions sentence, whatever its label.

## Step 2: The experiment as a pipeline

### 1. Inputs

20 sentences per emotion, written by Claude (`10_generate_stimuli.py`, stored in `stimuli.json["emotions"]`), for 6 emotions (joy, admiration, optimism, sadness, anger, fear). Each is one sentence about "Alex" that shows the emotion without naming it.

- *Weakness:* small set, all in the same style.

### 2. Model and layer

GPT-2 Medium, residual stream (`hook_resid_post`) at layer 14.

- *Weaknesses:*
  - GPT-2 Medium is a small model and not a chat model.
  - Only one layer (14).

### 3. Vector

For each emotion, the average last-token activation of its 20 sentences, minus the average of all 120 sentences (`emotion_extraction/02_extract_emotion_vectors.py`).

- *Weakness:* there are no neutral sentences, so a vector shows how an emotion differs from the other 5 emotions, not from "no emotion."

### 4. Measurement

**4.1 Outside data (GoEmotions).** Use `goemotions_validation/goemotions_100_per_emotion.csv`, a filtered copy of GoEmotions (Reddit comments). Filter rules (the code that applied them is not in this repo, so this is the record):

- only comments with a single label;
- only comments where raters agreed on that label;
- very short comments removed (shortest kept: 4 words);
- up to 100 comments per label; only the 6 labels that match my vectors are used.

For each sentence, subtract the dataset average, then compute the cosine similarity with each of the 6 vectors. Score: % of sentences where the correct emotion is highest.

- *Weakness:* if the score is low, I can't tell whether the vectors miss the emotion or the Reddit style is just too different.

**4.2 Our dataset.** For each of my 120 Alex sentences, subtract the average of all 120 (as with GoEmotions). Then check which of the 6 vectors has the highest cosine similarity. Score: % of sentences where it's the correct emotion.

- *Weakness:* only a sanity check, because the vectors came from these sentences.

**4.3 Held-out sentences.** The same score on 5 new Alex sentences per emotion, not used to build the vectors (`stimuli.json["emotions_heldout"]`, generated the same way as the inputs).

- *Weakness:* only 5 per emotion, so the % is rough.

**4.4 Steering output.** Add each emotion vector at layer 14, scaled to length 105 (the same `STEER_STRENGTH` as in `08_steered_decay_experiment.py`; the layer-14 residual norm is about 236). Controls: no steering, and a random vector of the same length (105). Two tasks:

1. Finish neutral sentence starts. I guess the emotion with the labels hidden.
2. Answer the 1–100 rating prompt for each of the 6 emotions.

- *Weaknesses:*
  - I'm the only judge.
  - Strength 105 may break the text.
  - GPT-2 may not answer the rating prompt with a number.

### 5. Output

1. A table of % correct for each emotion, for our dataset, the held-out sentences and GoEmotions (chance = 17%).
2. My blind-guess accuracy on the steered texts, plus a few example texts.
3. A 6×6 rating table (rows: emotion added, columns: emotion asked about), compared with no steering.
