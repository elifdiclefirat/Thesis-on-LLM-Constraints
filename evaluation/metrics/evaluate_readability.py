from pathlib import Path
import csv
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "results/readability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONSTRAINTS = [
    "e",
    "et",
    "eta",
    "etao",
    "etaoi",
    "etaoin",
]

PASSAGES = [
    "short",
    "medium",
    "long",
]


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize_words(text):
    """
    Tokenize words using the same general word definition
    used in the other thesis analysis scripts.
    """

    return re.findall(
        r"\b[\w’'-]+\b",
        text
    )


def split_sentences(text):
    """
    Approximate sentence segmentation.

    A sentence is assumed to end with '.', '!' or '?'
    followed by whitespace or the end of the text.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# SYLLABLE COUNTING
# ============================================================

def count_syllables(word):
    """
    Estimate the number of syllables in an English word.

    This is a rule-based approximation based on vowel groups.
    It is designed for corpus-level readability analysis,
    not linguistic phonetic transcription.
    """

    word = word.lower()

    # Remove non-alphabetic characters
    word = re.sub(
        r"[^a-z]",
        "",
        word
    )

    if not word:
        return 0

    # Very short words
    if len(word) <= 3:
        return 1

    # Count vowel groups
    vowels = "aeiouy"

    syllables = 0
    previous_was_vowel = False

    for character in word:

        is_vowel = character in vowels

        if is_vowel and not previous_was_vowel:
            syllables += 1

        previous_was_vowel = is_vowel

    # Silent final 'e'
    if (
        word.endswith("e")
        and syllables > 1
        and not word.endswith(
            ("le", "ye")
        )
    ):
        syllables -= 1

    # Words ending in consonant + "ed"
    if (
        word.endswith("ed")
        and len(word) > 3
        and word[-3] not in vowels
        and syllables > 1
    ):
        syllables -= 1

    # Ensure minimum of one syllable
    return max(1, syllables)


# ============================================================
# FLESCH READING EASE
# ============================================================

def calculate_flesch(text):
    """
    Calculate Flesch Reading Ease.

    Formula:

        FRE = 206.835
              - 1.015 * (words / sentences)
              - 84.6 * (syllables / words)
    """

    words = tokenize_words(text)
    sentences = split_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)

    if word_count == 0:
        return 0.0, 0, 0, 0

    if sentence_count == 0:
        sentence_count = 1

    syllable_count = sum(
        count_syllables(word)
        for word in words
    )

    flesch = (
        206.835
        - 1.015 * (
            word_count /
            sentence_count
        )
        - 84.6 * (
            syllable_count /
            word_count
        )
    )

    return (
        flesch,
        word_count,
        sentence_count,
        syllable_count,
    )


# ============================================================
# FILE LOADING
# ============================================================

def load_text(file_path):
    """
    Load UTF-8 text file.
    """

    return file_path.read_text(
        encoding="utf-8"
    ).strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("READABILITY EVALUATION")
    print("=" * 70)

    print(
        "Metric: Flesch Reading Ease"
    )

    print(
        f"Reference directory: "
        f"{REFERENCE_DIR}"
    )

    print(
        f"Generated directory: "
        f"{GENERATED_DIR}"
    )

    results = []

    # ========================================================
    # EVALUATE ALL CONDITIONS
    # ========================================================

    for constraint in CONSTRAINTS:

        print("\n" + "-" * 70)
        print(
            f"CONSTRAINT: {constraint}"
        )
        print("-" * 70)

        for passage in PASSAGES:

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            reference_file = (
                REFERENCE_DIR /
                f"passage_{passage}.txt"
            )

            # ------------------------------------------------
            # Generated
            # ------------------------------------------------

            generated_file = (
                GENERATED_DIR /
                constraint /
                f"passage_{passage}.txt"
            )

            if not reference_file.exists():

                print(
                    f"WARNING: Missing reference: "
                    f"{reference_file}"
                )

                continue

            if not generated_file.exists():

                print(
                    f"WARNING: Missing generated: "
                    f"{generated_file}"
                )

                continue

            reference_text = load_text(
                reference_file
            )

            generated_text = load_text(
                generated_file
            )

            # ------------------------------------------------
            # Calculate reference readability
            # ------------------------------------------------

            (
                original_fre,
                original_words,
                original_sentences,
                original_syllables,
            ) = calculate_flesch(
                reference_text
            )

            # ------------------------------------------------
            # Calculate generated readability
            # ------------------------------------------------

            (
                generated_fre,
                generated_words,
                generated_sentences,
                generated_syllables,
            ) = calculate_flesch(
                generated_text
            )

            # ------------------------------------------------
            # Difference
            # ------------------------------------------------

            difference = (
                generated_fre -
                original_fre
            )

            if original_fre != 0:

                relative_change = (
                    difference /
                    abs(original_fre)
                ) * 100

            else:

                relative_change = 0.0

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"\n{passage.capitalize()}"
            )

            print(
                f"  Original FRE : "
                f"{original_fre:.2f}"
            )

            print(
                f"  Generated FRE: "
                f"{generated_fre:.2f}"
            )

            print(
                f"  Difference   : "
                f"{difference:+.2f}"
            )

            print(
                f"  Relative     : "
                f"{relative_change:+.2f}%"
            )

            print(
                f"  Original     : "
                f"{original_words} words, "
                f"{original_sentences} sentences, "
                f"{original_syllables} syllables"
            )

            print(
                f"  Generated    : "
                f"{generated_words} words, "
                f"{generated_sentences} sentences, "
                f"{generated_syllables} syllables"
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            results.append({

                "Constraint":
                    constraint,

                "Passage":
                    passage,

                "Original_Words":
                    original_words,

                "Generated_Words":
                    generated_words,

                "Original_Sentences":
                    original_sentences,

                "Generated_Sentences":
                    generated_sentences,

                "Original_Syllables":
                    original_syllables,

                "Generated_Syllables":
                    generated_syllables,

                "Original_Flesch":
                    original_fre,

                "Generated_Flesch":
                    generated_fre,

                "Difference":
                    difference,

                "Relative_Change_Percent":
                    relative_change,
            })


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        OUTPUT_DIR /
        "readability_results.csv"
    )

    fieldnames = [
        "Constraint",
        "Passage",
        "Original_Words",
        "Generated_Words",
        "Original_Sentences",
        "Generated_Sentences",
        "Original_Syllables",
        "Generated_Syllables",
        "Original_Flesch",
        "Generated_Flesch",
        "Difference",
        "Relative_Change_Percent",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "READABILITY EVALUATION COMPLETED"
    )
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
