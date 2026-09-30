"""Reproduce the Shakespeare character-level Markov experiment for the poster."""

import json
import random
from collections import defaultdict
from pathlib import Path


TEXT_FILE = Path(__file__).with_name("shakespeare.txt")
CONTEXT_LENGTHS = range(1, 5)
SAMPLE_COUNT = 10
SAMPLE_LENGTH = 10_000
WINDOW_LENGTH = 10
SEED_TEXT = "romeo:"


class MarkovCharacterModel:
    def __init__(self, context_length):
        self.context_length = context_length
        self.transitions = defaultdict(list)

    def train(self, text):
        k = self.context_length
        for i in range(len(text) - k):
            context = tuple(text[i : i + k])
            next_character = text[i + k]
            self.transitions[context].append(next_character)

    def generate(self, length, seed_text):
        k = self.context_length
        possible_starts = list(self.transitions)
        seed = tuple(seed_text[-k:])
        context = seed if len(seed) == k and seed in self.transitions else random.choice(possible_starts)
        output = list(context)

        while len(output) < length:
            if context in self.transitions:
                output.append(random.choice(self.transitions[context]))
                context = tuple(output[-k:])
            else:
                context = random.choice(possible_starts)
                output.extend(context)

        return "".join(output[:length])


def windows(text, size):
    return (text[i : i + size] for i in range(len(text) - size + 1))


def main():
    if not TEXT_FILE.exists():
        raise SystemExit(
            f"Could not find {TEXT_FILE.name}. Put it in the same folder as this script."
        )

    source = TEXT_FILE.read_text(encoding="utf-8").lower()
    source_windows = set(windows(source, WINDOW_LENGTH))
    print(f"Corpus: {len(source):,} characters; {len(set(source))} distinct characters")
    print(f"Novelty means a generated {WINDOW_LENGTH}-character window is absent from the source.\n")

    for k in CONTEXT_LENGTHS:
        model = MarkovCharacterModel(k)
        model.train(source)
        novelty_rates = []

        for repetition in range(SAMPLE_COUNT):
            random.seed(8000 + k * 100 + repetition)
            sample = model.generate(SAMPLE_LENGTH, SEED_TEXT)
            sample_windows = list(windows(sample, WINDOW_LENGTH))
            novel = sum(window not in source_windows for window in sample_windows)
            novelty_rates.append(novel / len(sample_windows))

        mean = sum(novelty_rates) / len(novelty_rates)
        standard_deviation = (
            sum((rate - mean) ** 2 for rate in novelty_rates) / len(novelty_rates)
        ) ** 0.5
        # Print one separate example so the reader can see what the model wrote.
        random.seed(9000 + k)
        example = model.generate(240, SEED_TEXT).replace("\n", " ")
        print(
            json.dumps(
                {
                    "context_length": k,
                    "mean_novel_windows_percent": round(mean * 100, 2),
                    "standard_deviation_percentage_points": round(
                        standard_deviation * 100, 2
                    ),
                }
            )
        )
        print(f"Example passage (context length {k}):\n{example}\n")


if __name__ == "__main__":
    main()
