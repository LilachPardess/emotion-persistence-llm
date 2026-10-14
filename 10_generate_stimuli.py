"""
Generate one-sentence "Alex" emotion stories with a Claude model and store
them in stimuli.json under --output-key (default "emotions"), replacing that
section (other keys are untouched).

For each emotion, the model is asked --n times for a single
sentence that shows the emotion without naming it. Each sentence is set in
one topic from stimuli.json["topics"] (commute, job interview, ...), taken in
order, so the scenes vary and every emotion gets the same topics: sentence i
of every emotion shares a topic, and topic alone can't tell the emotions
apart. There are 25 topics, so 20 training + 5 held-out sentences use each
topic once per emotion. The topic order is saved under "<output-key>_topics".

Overused words: for each emotion separately, every accepted sentence's
content words (stop words like "she" and "the" skipped, word forms grouped by
Porter stem, so "trembled" and "trembling" count as one) are counted. A word
already in --max-word-count sentences of that emotion goes on that emotion's
avoid list: later prompts for the emotion name it, and a sentence that uses
it anyway is rejected. Early prompts carry no avoid list.

When writing to a key other than "emotions", sentences already in "emotions"
are rejected, so a held-out set never repeats a sentence the vectors were
built from, and its topics continue the cycle where "emotions" stopped.

Usage:
    pip install anthropic python-dotenv nltk scikit-learn
    echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
    python 10_generate_stimuli.py                                        # 20 per emotion -> "emotions"
    python 10_generate_stimuli.py --output-key emotions_heldout --n 5    # held-out set
"""
import argparse
import json
import re
from collections import Counter

from anthropic import Anthropic
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STIMULI_PATH = "stimuli.json"
TRAINING_KEY = "emotions"
MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 100
TEMPERATURE = 1.0
WORD_LIMIT = 20  # sentences must be strictly under this many words
MAX_SHORTEN_ROUNDS = 2
MAX_ATTEMPTS_PER_SENTENCE = 20

EMOTIONS = ["joy", "admiration", "optimism", "sadness", "anger", "fear"]

# Word-start stems; a sentence matching its emotion's stem is regenerated.
FORBIDDEN_STEMS = {
    "joy": r"joy",
    "admiration": r"admir",
    "optimism": r"optimis",
    "sadness": r"sad",
    "anger": r"ang(er|ry|rily)",
    "fear": r"fear",
}

STOP_WORDS = ENGLISH_STOP_WORDS | {"alex"}
STEMMER = PorterStemmer()

PROMPT_TEMPLATE = """You are an AI generating a precise linguistic dataset. You must output exactly ONE sentence, and nothing else. No introductions, no formatting, no quotation marks.
Write a short, engaging 1-sentence story (under 20 words) about a character named Alex who is experiencing a profound sense of {emotion}.
The story takes place during this situation: {topic}.
Constraints:

Describe the situation and how Alex reacts.
CRITICAL: Do not use the word "{emotion}" or any of its direct synonyms anywhere in the text.
{avoid_line}Output the raw sentence immediately."""

AVOID_LINE = "Avoid these words, which are already overused: {words}.\n"

SHORTEN_TEMPLATE = (
    "That sentence has {n} words. Rewrite it in under {limit} words, keeping "
    "all the same constraints. Output only the sentence."
)


def clean(text):
    return text.strip().strip('"\u201c\u201d\'').strip()


def content_words(sentence):
    """{stem: word as written} for the sentence's content words."""
    words = {}
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower()):
        word = word.removesuffix("'s")
        if word not in STOP_WORDS and "'" not in word and len(word) > 2:
            words.setdefault(STEMMER.stem(word), word)
    return words


def is_valid(sentence, emotion, existing, overused):
    if not sentence or sentence in existing:
        return False
    if len(sentence.split()) >= WORD_LIMIT:
        return False
    if re.search(rf"\b{FORBIDDEN_STEMS[emotion]}", sentence, re.IGNORECASE):
        return False
    if overused.keys() & content_words(sentence).keys():
        return False
    return True


def ask(client, messages):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=messages,
    )
    return clean(response.content[0].text)


def generate_sentence(client, emotion, topic, avoid_words):
    avoid_line = AVOID_LINE.format(words=", ".join(avoid_words)) if avoid_words else ""
    prompt = PROMPT_TEMPLATE.format(
        emotion=emotion, topic=topic.replace("_", " "), avoid_line=avoid_line
    )
    messages = [{"role": "user", "content": prompt}]
    sentence = ask(client, messages)
    for _ in range(MAX_SHORTEN_ROUNDS):
        n_words = len(sentence.split())
        if n_words < WORD_LIMIT:
            break
        messages += [
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": SHORTEN_TEMPLATE.format(n=n_words, limit=WORD_LIMIT)},
        ]
        sentence = ask(client, messages)
    return sentence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-key", default=TRAINING_KEY,
                        help=f"stimuli.json key to write (default: {TRAINING_KEY})")
    parser.add_argument("--n", type=int, default=20,
                        help="sentences per emotion (default: 20)")
    parser.add_argument("--max-word-count", type=int, default=3,
                        help="max sentences per emotion that may share a content "
                             "word (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_key, n_per_emotion = args.output_key, args.n

    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)
    # Sentences used to build the vectors; a held-out set must not repeat them.
    training = {} if output_key == TRAINING_KEY else stimuli.get(TRAINING_KEY, {})
    # Same topic order for every emotion; a held-out set continues the cycle.
    all_topics = stimuli["topics"]
    start = len(next(iter(training.values()), []))
    topics = [all_topics[(start + i) % len(all_topics)] for i in range(n_per_emotion)]
    stimuli[f"{output_key}_topics"] = topics

    load_dotenv()
    client = Anthropic()
    stimuli[output_key] = {}

    for emotion in EMOTIONS:
        sentences = []
        stem_counts = Counter()
        overused = {}  # stem -> word as first written, for this emotion only
        for i, topic in enumerate(topics):
            for attempt in range(1, MAX_ATTEMPTS_PER_SENTENCE + 1):
                sentence = generate_sentence(client, emotion, topic, sorted(overused.values()))
                if is_valid(sentence, emotion, sentences + training.get(emotion, []), overused):
                    break
                print(f"  [{emotion} {i+1}] rejected attempt {attempt}: {sentence!r}")
            else:
                raise RuntimeError(
                    f"No valid sentence for {emotion!r} after "
                    f"{MAX_ATTEMPTS_PER_SENTENCE} attempts"
                )
            sentences.append(sentence)
            print(f"[{emotion} {i+1:2d}/{n_per_emotion} {topic}] {sentence}")
            for stem, word in content_words(sentence).items():
                stem_counts[stem] += 1
                if stem_counts[stem] >= args.max_word_count:
                    overused.setdefault(stem, word)

        print(f"  {emotion} avoid list: {', '.join(sorted(overused.values())) or '(none)'}")

        stimuli[output_key][emotion] = sentences
        with open(STIMULI_PATH, "w") as f:
            json.dump(stimuli, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"\nSaved {len(EMOTIONS)} x {n_per_emotion} sentences to "
          f"{STIMULI_PATH} under '{output_key}'.")


if __name__ == "__main__":
    main()
