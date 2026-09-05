from pathlib import Path
import argparse
import csv
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

CONSTRAINTS = {
    "e": ["e"],
    "et": ["e", "t"],
    "eta": ["e", "t", "a"],
    "etao": ["e", "t", "a", "o"],
    "etaoi": ["e", "t", "a", "o", "i"],
    "etaoin": ["e", "t", "a", "o", "i", "n"],
}

PASSAGES = [
    "passage_short.txt",
    "passage_medium.txt",
    "passage_long.txt",
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Check lexical constraint compliance "
                "and structural statistics of generated passages."
)

parser.add_argument(
    "--constraint",
    required=True,
    choices=CONSTRAINTS.keys(),
    help="Constraint level to check."
)

args = parser.parse_args()

constraint_name = args.constraint
forbidden_letters = CONSTRAINTS[constraint_name]


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = RESULTS_ROOT / constraint_name

if not RESULTS_DIR.exists():
    raise FileNotFoundError(
        f"Results directory does not exist: {RESULTS_DIR}"
    )


# ============================================================
# TEXT STATISTICS
# ============================================================

def count_words(text):
    """
    Count word-like units using the same rule
    used during passage preparation.
    """
    return len(
        re.findall(
            r"\b[\w’'-]+\b",
            text
        )
    )


def count_sentences(text):
    """
    Count sentences using the same sentence
    segmentation rule used for corpus inspection.
    """

    if not text:
        return 0

    sentences = re.findall(
        r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<!\bSt)"
        r"[.!?]+"
        r"(?:[\"”»\]]+)?"
        r"(?=\s+[A-Z\"“«]|$)",
        text
    )

    return len(sentences)


def count_paragraphs(text):
    """
    Count paragraphs based on blank-line boundaries
    in the generated TXT file.
    """

    if not text.strip():
        return 0

    paragraphs = re.split(
        r"\n\s*\n",
        text.strip()
    )

    return len([
        paragraph
        for paragraph in paragraphs
        if paragraph.strip()
    ])


# ============================================================
# COMPLIANCE CHECK
# ============================================================

def count_forbidden_letters(text, forbidden):
    """
    Count occurrences of each forbidden letter,
    case-insensitively.
    """

    text_lower = text.lower()

    return {
        letter: text_lower.count(letter)
        for letter in forbidden
    }


# ============================================================
# MAIN CHECK
# ============================================================

print("=" * 70)
print("LEXICAL CONSTRAINT COMPLIANCE CHECK")
print("=" * 70)

print(f"Constraint:        {constraint_name}")
print(f"Forbidden letters: {forbidden_letters}")
print(f"Results directory: {RESULTS_DIR}")

print("=" * 70)


results = []


for passage_name in PASSAGES:

    output_file = RESULTS_DIR / passage_name

    if not output_file.exists():

        print(
            f"\nWARNING: Missing output: "
            f"{output_file}"
        )

        continue

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()


    # --------------------------------------------------------
    # Structural statistics
    # --------------------------------------------------------

    total_characters = len(text)
    total_words = count_words(text)
    total_paragraphs = count_paragraphs(text)
    total_sentences = count_sentences(text)


    # --------------------------------------------------------
    # Constraint compliance
    # --------------------------------------------------------

    forbidden_counts = count_forbidden_letters(
        text,
        forbidden_letters
    )

    total_forbidden = sum(
        forbidden_counts.values()
    )

    compliant = total_forbidden == 0


    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"PASSAGE: {passage_name}")
    print("-" * 70)

    print(
        f"Characters : {total_characters}"
    )

    print(
        f"Words      : {total_words}"
    )

    print(
        f"Paragraphs : {total_paragraphs}"
    )

    print(
        f"Sentences  : {total_sentences}"
    )

    print("\nForbidden-letter occurrences:")

    for letter, count in forbidden_counts.items():

        print(
            f"  '{letter}': {count}"
        )

    print(
        f"\nCompliance: "
        f"{'PASS' if compliant else 'FAIL'}"
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Constraint": constraint_name,

        "Passage": passage_name,

        "Total_Characters": total_characters,

        "Total_Words": total_words,

        "Total_Paragraphs": total_paragraphs,

        "Total_Sentences": total_sentences,

        **{
            f"Forbidden_{letter}": count
            for letter, count in forbidden_counts.items()
        },

        "Total_Forbidden": total_forbidden,

        "Compliant": compliant,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

if results:

    output_csv = (
        RESULTS_DIR /
        "compliance.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        output_csv,
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


    print("\n" + "=" * 70)
    print("COMPLIANCE CHECK COMPLETED")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{output_csv}"
    )

else:

    print("\nNo generated passages found.")


print("=" * 70)
