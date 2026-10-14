# Stimuli Generation

## Goal
We built a set of short sentences, each showing one emotion without naming it. We use them to extract an emotion vector for each emotion.

## Emotions
Joy, admiration, optimism, sadness, anger and fear.

## Method
For each emotion, we asked Claude (`claude-sonnet-4-6`, temperature 1.0) to write a one-sentence story about a character named Alex who feels a profound sense of that emotion. Each sentence is set in a different everyday situation (a "topic").

**Prompt:**

```
You are an AI generating a precise linguistic dataset. You must output exactly ONE sentence, and nothing else. No introductions, no formatting, no quotation marks.
Write a short, engaging 1-sentence story (under 20 words) about a character named Alex who is experiencing a profound sense of {emotion}.
The story takes place during this situation: {topic}.
Constraints:

Describe the situation and how Alex reacts.
CRITICAL: Do not use the word "{emotion}" or any of its direct synonyms anywhere in the text.
Avoid these words, which are already overused: {words}.
Output the raw sentence immediately.
```

The "Avoid these words" line is only included once the emotion has overused words. The first prompts for each emotion don't have it.

## Dataset size
- **Training set:** 20 sentences per emotion (120 in total), used to build the vectors.
- **Held-out set:** 5 sentences per emotion (30 in total), not used to build the vectors and used only for testing. A held-out sentence can't be identical to a training sentence.

## Quality controls
1. **The emotion word isn't used.** The prompt tells the model not to use the emotion word or its synonyms. The code then rejects any sentence that contains the word's stem (for example *sad*, *angry* or *fearful* for those emotions) and asks for a new one. Synonyms (for example *dread*, *elated*) are only discouraged by the prompt, not checked by the code.
2. **Topics are varied and matched across emotions.** There are 25 topics (commute, job interview, storm, hospital visit, laundry, etc.). Sentence *i* of every emotion uses the same topic, so the topic alone can't tell the emotions apart. The 20 training and 5 held-out sentences together use each topic exactly once per emotion.
3. **Words don't repeat too often.** For each emotion, we count the content words in every accepted sentence. We skip stop words, and different forms of a word count as one (*trembled* and *trembling* are the same). Once a word appears in 3 sentences of the same emotion, it goes on that emotion's avoid list. Later prompts tell the model to avoid it, and any sentence that still uses it is rejected.
4. **Length.** Sentences must be under 20 words. If a sentence is too long, the model is asked to shorten it (up to 2 times). If it is still too long, it is rejected.
5. **No duplicates.** A sentence identical to an earlier one is rejected. The model gets up to 20 attempts per sentence.

## Examples
| Emotion | Topic | Sentence |
|---|---|---|
| Joy | commute | Alex grinned uncontrollably on the crowded subway, replaying the morning's surprise promotion in his mind over and over. |
| Sadness | commute | Alex pressed his forehead against the cold bus window, watching blurred streetlights mirror his hollow chest. |
| Fear | commute | Alex's hands trembled as the subway lights died, leaving him alone in suffocating darkness. |

## Limitations
- Alex's gender isn't fixed: most sentences use "he", but some use "them".

## Output
All sentences are saved in `stimuli.json` under `emotions` (training) and `emotions_heldout`. The topic order is saved under `emotions_topics` and `emotions_heldout_topics`.

Script: `10_generate_stimuli.py`.
