from pathlib import Path
import re
import math
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results/d"
OUTPUT_DIR = PROJECT_ROOT / "results/readability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "d_readability_results.csv"


# ============================================================
# Configuration
# ============================================================

CONSTRAINT = "d"

PASSAGES = [
    "short",
    "medium",
    "long",
]


# ============================================================
# Tokenization
# ============================================================

def tokenize_words(text):
    return re.findall(r"\b[\w’'-]+\b", text)


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


# ============================================================
# Syllable counting
# ============================================================

def count_syllables(word):
    """
    Estimate the number of syllables in an English word using
    a rule-based approach.
    """

    word = word.lower()

    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)

    if not word:
        return 0

    vowels = "aeiouy"

    syllables = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels

        if is_vowel and not previous_was_vowel:
            syllables += 1

        previous_was_vowel = is_vowel

    # Silent final "e"
    if word.endswith("e") and syllables > 1:
        syllables -= 1

    # Words ending in consonant + "le"
    if (
        len(word) > 2
        and word.endswith("le")
        and word[-3] not in vowels
    ):
        syllables += 1

    # Words ending in consonant + "ed"
    if (
        len(word) > 2
        and word.endswith("ed")
        and word[-3] not in vowels
    ):
        syllables -= 1

    return max(1, syllables)


# ============================================================
# Readability
# ============================================================

def calculate_flesch_reading_ease(text):

    words = tokenize_words(text)

    sentences = split_sentences(text)

    # Remove empty sentences
    sentences = [
        sentence
        for sentence in sentences
        if sentence.strip()
    ]

    word_count = len(words)
    sentence_count = len(sentences)

    if word_count == 0 or sentence_count == 0:
        return {
            "words": word_count,
            "sentences": sentence_count,
            "syllables": 0,
            "flesch_reading_ease": math.nan,
        }

    syllable_count = sum(
        count_syllables(word)
        for word in words
    )

    flesch = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )

    return {
        "words": word_count,
        "sentences": sentence_count,
        "syllables": syllable_count,
        "flesch_reading_ease": flesch,
    }


# ============================================================
# Main evaluation
# ============================================================

results = []


for passage in PASSAGES:

    reference_file = (
        REFERENCE_DIR / f"passage_{passage}.txt"
    )

    generated_file = (
        GENERATED_DIR / f"passage_{passage}.txt"
    )

    print(f"Processing {CONSTRAINT} - {passage}...")

    # --------------------------------------------------------
    # Read reference
    # --------------------------------------------------------

    with open(
        reference_file,
        "r",
        encoding="utf-8"
    ) as f:
        reference_text = f.read()

    # --------------------------------------------------------
    # Read generated
    # --------------------------------------------------------

    with open(
        generated_file,
        "r",
        encoding="utf-8"
    ) as f:
        generated_text = f.read()

    # --------------------------------------------------------
    # Calculate readability
    # --------------------------------------------------------

    reference_metrics = calculate_flesch_reading_ease(
        reference_text
    )

    generated_metrics = calculate_flesch_reading_ease(
        generated_text
    )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    word_difference = (
        generated_metrics["words"]
        - reference_metrics["words"]
    )

    sentence_difference = (
        generated_metrics["sentences"]
        - reference_metrics["sentences"]
    )

    syllable_difference = (
        generated_metrics["syllables"]
        - reference_metrics["syllables"]
    )

    flesch_difference = (
        generated_metrics["flesch_reading_ease"]
        - reference_metrics["flesch_reading_ease"]
    )

    # --------------------------------------------------------
    # Relative change
    # --------------------------------------------------------

    reference_flesch = (
        reference_metrics["flesch_reading_ease"]
    )

    if reference_flesch != 0:
        relative_change = (
            flesch_difference
            / reference_flesch
        ) * 100
    else:
        relative_change = math.nan

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "Constraint": CONSTRAINT,
        "Passage": passage,

        "Original Words":
            reference_metrics["words"],

        "Generated Words":
            generated_metrics["words"],

        "Words Difference":
            word_difference,

        "Original Sentences":
            reference_metrics["sentences"],

        "Generated Sentences":
            generated_metrics["sentences"],

        "Sentences Difference":
            sentence_difference,

        "Original Syllables":
            reference_metrics["syllables"],

        "Generated Syllables":
            generated_metrics["syllables"],

        "Syllables Difference":
            syllable_difference,

        "Original Flesch":
            reference_metrics["flesch_reading_ease"],

        "Generated Flesch":
            generated_metrics["flesch_reading_ease"],

        "Flesch Difference":
            flesch_difference,

        "Relative Change Percent":
            relative_change,
    })


# ============================================================
# Save results
# ============================================================

df = pd.DataFrame(results)
from pathlib import Path
import re
import math
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results/d"
OUTPUT_DIR = PROJECT_ROOT / "results/readability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "d_readability_results.csv"


# ============================================================
# Configuration
# ============================================================

CONSTRAINT = "d"

PASSAGES = [
    "short",
    "medium",
    "long",
]


# ============================================================
# Tokenization
# ============================================================

def tokenize_words(text):
    return re.findall(r"\b[\w’'-]+\b", text)


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


# ============================================================
# Syllable counting
# ============================================================

def count_syllables(word):
    """
    Estimate the number of syllables in an English word using
    a rule-based approach.
    """

    word = word.lower()

    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)

    if not word:
        return 0

    vowels = "aeiouy"

    syllables = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels

        if is_vowel and not previous_was_vowel:
            syllables += 1

        previous_was_vowel = is_vowel

    # Silent final "e"
    if word.endswith("e") and syllables > 1:
        syllables -= 1

    # Words ending in consonant + "le"
    if (
        len(word) > 2
        and word.endswith("le")
        and word[-3] not in vowels
    ):
        syllables += 1

    # Words ending in consonant + "ed"
    if (
        len(word) > 2
        and word.endswith("ed")
        and word[-3] not in vowels
    ):
        syllables -= 1

    return max(1, syllables)


# ============================================================
# Readability
# ============================================================

def calculate_flesch_reading_ease(text):

    words = tokenize_words(text)

    sentences = split_sentences(text)

    # Remove empty sentences
    sentences = [
        sentence
        for sentence in sentences
        if sentence.strip()
    ]

    word_count = len(words)
    sentence_count = len(sentences)

    if word_count == 0 or sentence_count == 0:
        return {
            "words": word_count,
            "sentences": sentence_count,
            "syllables": 0,
            "flesch_reading_ease": math.nan,
        }

    syllable_count = sum(
        count_syllables(word)
        for word in words
    )

    flesch = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )

    return {
        "words": word_count,
        "sentences": sentence_count,
        "syllables": syllable_count,
        "flesch_reading_ease": flesch,
    }


# ============================================================
# Main evaluation
# ============================================================

results = []


for passage in PASSAGES:

    reference_file = (
        REFERENCE_DIR / f"passage_{passage}.txt"
    )

    generated_file = (
        GENERATED_DIR / f"passage_{passage}.txt"
    )

    print(f"Processing {CONSTRAINT} - {passage}...")

    # --------------------------------------------------------
    # Read reference
    # --------------------------------------------------------

    with open(
        reference_file,
        "r",
        encoding="utf-8"
    ) as f:
        reference_text = f.read()

    # --------------------------------------------------------
    # Read generated
    # --------------------------------------------------------

    with open(
        generated_file,
        "r",
        encoding="utf-8"
    ) as f:
        generated_text = f.read()

    # --------------------------------------------------------
    # Calculate readability
    # --------------------------------------------------------

    reference_metrics = calculate_flesch_reading_ease(
        reference_text
    )

    generated_metrics = calculate_flesch_reading_ease(
        generated_text
    )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    word_difference = (
        generated_metrics["words"]
        - reference_metrics["words"]
    )

    sentence_difference = (
        generated_metrics["sentences"]
        - reference_metrics["sentences"]
    )

    syllable_difference = (
        generated_metrics["syllables"]
        - reference_metrics["syllables"]
    )

    flesch_difference = (
        generated_metrics["flesch_reading_ease"]
        - reference_metrics["flesch_reading_ease"]
    )

    # --------------------------------------------------------
    # Relative change
    # --------------------------------------------------------

    reference_flesch = (
        reference_metrics["flesch_reading_ease"]
    )

    if reference_flesch != 0:
        relative_change = (
            flesch_difference
            / reference_flesch
        ) * 100
    else:
        relative_change = math.nan

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "Constraint": CONSTRAINT,
        "Passage": passage,

        "Original Words":
            reference_metrics["words"],

        "Generated Words":
            generated_metrics["words"],

        "Words Difference":
            word_difference,

        "Original Sentences":
            reference_metrics["sentences"],

        "Generated Sentences":
            generated_metrics["sentences"],

        "Sentences Difference":
            sentence_difference,

        "Original Syllables":
            reference_metrics["syllables"],

        "Generated Syllables":
            generated_metrics["syllables"],

        "Syllables Difference":
            syllable_difference,

        "Original Flesch":
            reference_metrics["flesch_reading_ease"],

        "Generated Flesch":
            generated_metrics["flesch_reading_ease"],

        "Flesch Difference":
            flesch_difference,

        "Relative Change Percent":
            relative_change,
    })


# ============================================================
# Save results
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Terminal output
# ============================================================

print()
print("Evaluation complete.")
print(f"Output saved to: {OUTPUT_FILE}")
print()

for _, row in df.iterrows():

    print("=" * 60)

    print(
        f"Constraint: {row['Constraint']} | "
        f"Passage: {row['Passage']}"
    )

    print("=" * 60)

    print(
        f"Words:        "
        f"{row['Original Words']} -> "
        f"{row['Generated Words']} "
        f"(Δ {row['Words Difference']})"
    )

    print(
        f"Sentences:    "
        f"{row['Original Sentences']} -> "
        f"{row['Generated Sentences']} "
        f"(Δ {row['Sentences Difference']})"
    )

    print(
        f"Syllables:    "
        f"{row['Original Syllables']} -> "
        f"{row['Generated Syllables']} "
        f"(Δ {row['Syllables Difference']})"
    )

    print(
        f"Flesch:       "
        f"{row['Original Flesch']:.6f} -> "
        f"{row['Generated Flesch']:.6f}"
    )

    print(
        f"Flesch Δ:     "
        f"{row['Flesch Difference']:.6f}"
    )

    print(
        f"Relative Δ:   "
        f"{row['Relative Change Percent']:.2f}%"
    )

    print()


print("All results saved successfully.")
