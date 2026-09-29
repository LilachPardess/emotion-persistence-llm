"""
Generate one-sentence "Alex" emotion stories with a Claude model and store
them in stimuli.json under "generated_emotions" (existing keys are untouched).

For each emotion, the model is asked N_PER_EMOTION times for a single
sentence that conveys the emotion only through physical sensations and
internal thoughts, without naming it.

Usage:
    pip install anthropic python-dotenv
    echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
    python 10_generate_stimuli.py
"""
import json
import re

from anthropic import Anthropic
from dotenv import load_dotenv

STIMULI_PATH = "stimuli.json"
OUTPUT_KEY = "generated_emotions"
MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 100
TEMPERATURE = 1.0
N_PER_EMOTION = 20
WORD_LIMIT = 20  # sentences must be strictly under this many words
MAX_SHORTEN_ROUNDS = 2
MAX_ATTEMPTS_PER_SENTENCE = 10

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

PROMPT_TEMPLATE = """You are an AI generating a precise linguistic dataset. You must output exactly ONE sentence, and nothing else. No introductions, no formatting, no quotation marks.
Write a short, engaging 1-sentence story (under 20 words) about a character named Alex who is experiencing a profound sense of {emotion}.
Constraints:

Describe the situation and how Alex reacts.
Express the feeling entirely through Alex's physical sensations and internal thoughts.
CRITICAL: Do not use the word "{emotion}" or any of its direct synonyms anywhere in the text.
Output the raw sentence immediately."""

SHORTEN_TEMPLATE = (
    "That sentence has {n} words. Rewrite it in under {limit} words, keeping "
    "all the same constraints. Output only the sentence."
)


def clean(text):
    return text.strip().strip('"\u201c\u201d\'').strip()


def is_valid(sentence, emotion, existing):
    if not sentence or sentence in existing:
        return False
    if len(sentence.split()) >= WORD_LIMIT:
        return False
    if re.search(rf"\b{FORBIDDEN_STEMS[emotion]}", sentence, re.IGNORECASE):
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


def generate_sentence(client, emotion):
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(emotion=emotion)}]
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


def main():
    load_dotenv()
    client = Anthropic()

    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)
    stimuli[OUTPUT_KEY] = {}

    for emotion in EMOTIONS:
        sentences = []
        for i in range(N_PER_EMOTION):
            for attempt in range(1, MAX_ATTEMPTS_PER_SENTENCE + 1):
                sentence = generate_sentence(client, emotion)
                if is_valid(sentence, emotion, sentences):
                    break
                print(f"  [{emotion} {i+1}] rejected attempt {attempt}: {sentence!r}")
            else:
                raise RuntimeError(
                    f"No valid sentence for {emotion!r} after "
                    f"{MAX_ATTEMPTS_PER_SENTENCE} attempts"
                )
            sentences.append(sentence)
            print(f"[{emotion} {i+1:2d}/{N_PER_EMOTION}] {sentence}")

        stimuli[OUTPUT_KEY][emotion] = sentences
        with open(STIMULI_PATH, "w") as f:
            json.dump(stimuli, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"\nSaved {len(EMOTIONS)} x {N_PER_EMOTION} sentences to "
          f"{STIMULI_PATH} under '{OUTPUT_KEY}'.")


if __name__ == "__main__":
    main()
