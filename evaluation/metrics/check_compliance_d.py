from pathlib import Path
import csv
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data" / "processed"
GENERATED_DIR = PROJECT_ROOT / "results" / "d"
OUTPUT_DIR = PROJECT_ROOT / "results" / "compliance"

OUTPUT_FILE = OUTPUT_DIR / "d_compliance.csv"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CONSTRAINT = "d"

PASSAGES = [
    "short",
    "medium",
    "long",
]


# ============================================================
# TEXT ANALYSIS
# ============================================================

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def count_paragraphs(text):
    paragraphs = [
        p.strip()
        for p in text.splitlines()
        if p.strip()
    ]

    return len(paragraphs)


def count_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentences = [
        s for s in sentences
        if s.strip()
    ]

    return len(sentences)


def count_words(text):
    words = re.findall(
        r"\b[\w’'-]+\b",
        text
    )

    return len(words)


def count_characters(text):
    return len(text)


def count_forbidden_characters(
    text,
    forbidden_character
):
    return text.lower().count(
        forbidden_character.lower()
    )


# ============================================================
# DERIVED METRICS
# ============================================================

def calculate_delta(
    original,
    generated
):
    """
    Percentage change relative to the original.
    """

    if original == 0:
        return 0.0

    return (
        (generated - original)
        / original
        * 100
    )


def calculate_expansion(
    original,
    generated
):
    """
    Generated / Original.
    """

    if original == 0:
        return 0.0

    return generated / original


def calculate_violation_ratio(
    total_forbidden,
    generated_characters
):
    """
    Forbidden-character ratio relative to the
    generated output character count.
    """

    if generated_characters == 0:
        return 0.0

    return (
        total_forbidden
        / generated_characters
        * 100
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    print("=" * 80)
    print("Constraint Compliance Analysis")
    print("=" * 80)
    print(f"Constraint: {CONSTRAINT.upper()}")
    print()

    for passage in PASSAGES:

        reference_file = (
            REFERENCE_DIR
            / f"passage_{passage}.txt"
        )

        generated_file = (
            GENERATED_DIR
            / f"passage_{passage}.txt"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not reference_file.exists():

            print(
                f"[WARNING] Reference file not found: "
                f"{reference_file}"
            )

            continue

        if not generated_file.exists():

            print(
                f"[WARNING] Generated file not found: "
                f"{generated_file}"
            )

            continue

        # ----------------------------------------------------
        # Load texts
        # ----------------------------------------------------

        original_text = load_text(
            reference_file
        )

        generated_text = load_text(
            generated_file
        )

        # ----------------------------------------------------
        # Original passage metrics
        # ----------------------------------------------------

        original_paragraphs = count_paragraphs(
            original_text
        )

        original_sentences = count_sentences(
            original_text
        )

        original_words = count_words(
            original_text
        )

        original_characters = count_characters(
            original_text
        )

        original_forbidden = (
            count_forbidden_characters(
                original_text,
                CONSTRAINT
            )
        )

        # ----------------------------------------------------
        # Generated passage metrics
        # ----------------------------------------------------

        generated_paragraphs = count_paragraphs(
            generated_text
        )

        generated_sentences = count_sentences(
            generated_text
        )

        generated_words = count_words(
            generated_text
        )

        generated_characters = count_characters(
            generated_text
        )

        generated_forbidden = (
            count_forbidden_characters(
                generated_text,
                CONSTRAINT
            )
        )

        # ----------------------------------------------------
        # Passage-level changes
        # ----------------------------------------------------

        paragraph_delta = calculate_delta(
            original_paragraphs,
            generated_paragraphs
        )

        sentence_delta = calculate_delta(
            original_sentences,
            generated_sentences
        )

        words_expansion = calculate_expansion(
            original_words,
            generated_words
        )

        words_delta = calculate_delta(
            original_words,
            generated_words
        )

        characters_delta = calculate_delta(
            original_characters,
            generated_characters
        )

        # ----------------------------------------------------
        # Constraint metrics
        # ----------------------------------------------------

        violation_ratio = calculate_violation_ratio(
            generated_forbidden,
            generated_characters
        )

        # ----------------------------------------------------
        # Store row
        # ----------------------------------------------------

        rows.append({
            "Passage": f"passage_{passage}.txt",

            "Paragraphs": generated_paragraphs,
            "Paragraph Delta": paragraph_delta,

            "Sentences": generated_sentences,
            "Sentence Delta": sentence_delta,

            "Words": generated_words,
            "Words Expansion": words_expansion,
            "Words Delta": words_delta,

            "Characters": generated_characters,
            "Characters Delta": characters_delta,

            "Total Forbidden": generated_forbidden,
            "Ratio": violation_ratio,

            "d Original": original_forbidden,
            "d Generated": generated_forbidden,
        })

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Passage: passage_{passage}.txt"
        )

        print(
            f"  Paragraphs: "
            f"{original_paragraphs} -> "
            f"{generated_paragraphs} "
            f"({paragraph_delta:.2f}%)"
        )

        print(
            f"  Sentences: "
            f"{original_sentences} -> "
            f"{generated_sentences} "
            f"({sentence_delta:.2f}%)"
        )

        print(
            f"  Words: "
            f"{original_words} -> "
            f"{generated_words} "
            f"(x{words_expansion:.2f}, "
            f"{words_delta:.2f}%)"
        )

        print(
            f"  Characters: "
            f"{original_characters} -> "
            f"{generated_characters} "
            f"({characters_delta:.2f}%)"
        )

        print(
            f"  Forbidden D: "
            f"{original_forbidden} -> "
            f"{generated_forbidden}"
        )

        print(
            f"  Violation ratio: "
            f"{violation_ratio:.2f}%"
        )

        print()

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "Passage",

        "Paragraphs",
        "Paragraph Delta",

        "Sentences",
        "Sentence Delta",

        "Words",
        "Words Expansion",
        "Words Delta",

        "Characters",
        "Characters Delta",

        "Total Forbidden",
        "Ratio",

        "d Original",
        "d Generated",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Results saved to:")
    print(OUTPUT_FILE)
    print("=" * 80)


if __name__ == "__main__":
    main()
