from pathlib import Path
import re
import csv

from docx import Document


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path("/netscratch/efirat/thesis")

SOURCE_DIR = BASE_DIR / "data/source"
PROCESSED_DIR = BASE_DIR / "data/processed"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STORY_OUTPUT_FILE = RESULTS_DIR / "story_inspection.csv"
PASSAGE_OUTPUT_FILE = RESULTS_DIR / "passage_inspection.csv"


# --------------------------------------------------
# Text processing helpers
# --------------------------------------------------

def normalize_text(text):
    """
    Clean whitespace while preserving the actual words.
    """
    return re.sub(r"\s+", " ", text).strip()


def count_words(text):
    """
    Count word-like units using the same rule used
    during passage preparation.
    """
    return len(
        re.findall(
            r"\b[\w’'-]+\b",
            text
        )
    )


def count_sentences(text):
    """
    Approximate sentence count using sentence-ending
    punctuation.

    This is used for corpus and passage inspection.
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
    in the processed TXT files.
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


def count_constraint_letters(text):
    """
    Count occurrences of the six letters used in the
    lexical constraint experiment.

    Counting is case-insensitive.
    """

    text_lower = text.lower()

    letters = [
        "e",
        "t",
        "a",
        "o",
        "i",
        "n",
    ]

    return {
        f"Count_{letter.upper()}":
            text_lower.count(letter)
        for letter in letters
    }


# --------------------------------------------------
# Original story inspection
# --------------------------------------------------

def inspect_story(file_path):
    """
    Inspect one DOCX story and return basic statistics.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = normalize_text(
            paragraph.text
        )

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    total_characters = len(full_text)
    total_words = count_words(full_text)
    total_paragraphs = len(paragraphs)
    total_sentences = count_sentences(
        full_text
    )

    return {
        "Story": file_path.stem,
        "Total_Characters": total_characters,
        "Total_Words": total_words,
        "Total_Paragraphs": total_paragraphs,
        "Total_Sentences": total_sentences,
    }


# --------------------------------------------------
# Processed passage inspection
# --------------------------------------------------

def inspect_passage(file_path):
    """
    Inspect one processed experimental passage.

    The original TXT file is only read.
    It is NOT modified.
    """

    text = file_path.read_text(
        encoding="utf-8"
    )

    return {
        "Passage": file_path.stem.replace(
            "passage_",
            ""
        ),

        "Total_Characters": len(text),

        "Total_Words": count_words(text),

        "Total_Paragraphs": count_paragraphs(
            text
        ),

        "Total_Sentences": count_sentences(
            text
        ),

        **count_constraint_letters(text),
    }


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

def save_csv(
    output_file,
    fieldnames,
    results
):
    """
    Save inspection results to CSV.
    """

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # ==================================================
    # ORIGINAL STORY
    # ==================================================

    story_files = sorted(
        SOURCE_DIR.glob("*.docx")
    )

    print("=" * 70)
    print("ORIGINAL STORY INSPECTION")
    print("=" * 70)

    story_results = []

    if not story_files:

        print(
            f"No DOCX files found in: "
            f"{SOURCE_DIR}"
        )

    else:

        for story_file in story_files:

            print(
                f"\nInspecting: "
                f"{story_file.name}"
            )

            stats = inspect_story(
                story_file
            )

            story_results.append(stats)

            print(
                f"  Characters : "
                f"{stats['Total_Characters']}"
            )

            print(
                f"  Words      : "
                f"{stats['Total_Words']}"
            )

            print(
                f"  Paragraphs : "
                f"{stats['Total_Paragraphs']}"
            )

            print(
                f"  Sentences  : "
                f"{stats['Total_Sentences']}"
            )


    story_fieldnames = [
        "Story",
        "Total_Characters",
        "Total_Words",
        "Total_Paragraphs",
        "Total_Sentences",
    ]

    save_csv(
        STORY_OUTPUT_FILE,
        story_fieldnames,
        story_results
    )


    # ==================================================
    # PROCESSED PASSAGES
    # ==================================================

    passage_files = [
        PROCESSED_DIR /
        "passage_short.txt",

        PROCESSED_DIR /
        "passage_medium.txt",

        PROCESSED_DIR /
        "passage_long.txt",
    ]

    print("\n" + "=" * 70)
    print("PROCESSED PASSAGE INSPECTION")
    print("=" * 70)

    passage_results = []

    for passage_file in passage_files:

        if not passage_file.exists():

            print(
                f"\nWARNING: File not found: "
                f"{passage_file}"
            )

            continue

        print(
            f"\nInspecting: "
            f"{passage_file.name}"
        )

        stats = inspect_passage(
            passage_file
        )

        passage_results.append(stats)

        print(
            f"  Characters : "
            f"{stats['Total_Characters']}"
        )

        print(
            f"  Words      : "
            f"{stats['Total_Words']}"
        )

        print(
            f"  Paragraphs : "
            f"{stats['Total_Paragraphs']}"
        )

        print(
            f"  Sentences  : "
            f"{stats['Total_Sentences']}"
        )

        print(
            "\n  Constraint-letter counts:"
        )

        print(
            f"    E : {stats['Count_E']}"
        )

        print(
            f"    T : {stats['Count_T']}"
        )

        print(
            f"    A : {stats['Count_A']}"
        )

        print(
            f"    O : {stats['Count_O']}"
        )

        print(
            f"    I : {stats['Count_I']}"
        )

        print(
            f"    N : {stats['Count_N']}"
        )


    passage_fieldnames = [
        "Passage",
        "Total_Characters",
        "Total_Words",
        "Total_Paragraphs",
        "Total_Sentences",
        "Count_E",
        "Count_T",
        "Count_A",
        "Count_O",
        "Count_I",
        "Count_N",
    ]

    save_csv(
        PASSAGE_OUTPUT_FILE,
        passage_fieldnames,
        passage_results
    )


    # ==================================================
    # Finished
    # ==================================================

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)

    print(
        f"\nStory results saved to:"
        f"\n{STORY_OUTPUT_FILE}"
    )

    print(
        f"\nPassage results saved to:"
        f"\n{PASSAGE_OUTPUT_FILE}"
    )

    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
