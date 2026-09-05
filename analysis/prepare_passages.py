from pathlib import Path
import re
import csv

from docx import Document


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path("/netscratch/efirat/thesis")
SOURCE_FILE = BASE_DIR / "data/source/A SCANDAL IN BOHEMIA.docx"

OUTPUT_DIR = BASE_DIR / "data/processed"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / "passage_lengths.csv"


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
    Count word-like units using the same rule as the
    original passage preparation code.
    """
    return len(re.findall(r"\b[\w’'-]+\b", text))


def split_sentences(text):
    """
    Split the story into complete sentences.

    A sentence boundary is assumed after '.', '!' or '?'
    when followed by whitespace or the end of the text.
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


# --------------------------------------------------
# Load story
# --------------------------------------------------

def load_story(file_path):
    """
    Read all non-empty DOCX paragraphs and combine them
    into one continuous story.
    """
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = normalize_text(paragraph.text)

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


# --------------------------------------------------
# Build passage
# --------------------------------------------------

def build_passage(sentences, target_words):
    """
    Starting from the beginning of the story, keep complete
    sentences until the cumulative word count reaches or
    exceeds the target.

    No sentence is cut in the middle.
    """

    selected = []
    cumulative_words = 0

    for sentence in sentences:

        selected.append(sentence)
        cumulative_words += count_words(sentence)

        if cumulative_words >= target_words:
            break

    passage = " ".join(selected)

    return passage, cumulative_words


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("PREPARING EXPERIMENTAL PASSAGES")
    print("=" * 70)

    print(f"\nSource: {SOURCE_FILE}")

    if not SOURCE_FILE.exists():
        print("\nERROR: Source file not found.")
        return

    # --------------------------------------------------
    # Load complete story
    # --------------------------------------------------

    story = load_story(SOURCE_FILE)

    total_words = count_words(story)

    print(f"Total story words: {total_words}")

    # --------------------------------------------------
    # Split story into sentences
    # --------------------------------------------------

    sentences = split_sentences(story)

    print(f"Sentences found: {len(sentences)}")

    # --------------------------------------------------
    # Experimental passage targets
    # --------------------------------------------------

    targets = {
        "short": 2000,
        "medium": 5000,
    }

    summary = []

    # --------------------------------------------------
    # Create Short and Medium passages
    # --------------------------------------------------

    for passage_name, target in targets.items():

        passage, actual_words = build_passage(
            sentences,
            target
        )

        output_file = (
            OUTPUT_DIR /
            f"passage_{passage_name}.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(passage)

        difference = actual_words - target

        summary.append({
            "Passage": passage_name,
            "Target_Words": target,
            "Actual_Words": actual_words,
            "Difference": difference,
            "Output_File": str(output_file),
        })

        print(
            f"{passage_name.capitalize():8s}: "
            f"Target = {target:5d} "
            f"-> Actual = {actual_words:5d} "
            f"(+{difference:3d}) "
            f"-> {output_file.name}"
        )

    # --------------------------------------------------
    # Create Long passage
    # --------------------------------------------------

    long_output = OUTPUT_DIR / "passage_long.txt"

    with open(
        long_output,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(story)

    summary.append({
        "Passage": "long",
        "Target_Words": total_words,
        "Actual_Words": total_words,
        "Difference": 0,
        "Output_File": str(long_output),
    })

    print(
        f"{'Long':8s}: "
        f"Full story "
        f"-> Actual = {total_words:5d} "
        f"-> {long_output.name}"
    )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    fieldnames = [
        "Passage",
        "Target_Words",
        "Actual_Words",
        "Difference",
        "Output_File",
    ]

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary)

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PASSAGES PREPARED SUCCESSFULLY")
    print("=" * 70)

    print("\nPassages saved to:")
    print(OUTPUT_DIR)

    print("\nSummary saved to:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
