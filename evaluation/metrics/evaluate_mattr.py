from pathlib import Path
import csv
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "results/mattr"

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

# Standard fixed window for MATTR
WINDOW_SIZE = 100


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Tokenize text using the same general word definition
    used in the thesis corpus inspection.

    Apostrophes and hyphens inside words are preserved.
    """
    return re.findall(
        r"\b[\w’'-]+\b",
        text.lower()
    )


# ============================================================
# MATTR
# ============================================================

def calculate_mattr(text, window_size=WINDOW_SIZE):
    """
    Calculate Moving-Average Type-Token Ratio (MATTR).

    MATTR is calculated as the average TTR across all
    overlapping windows of a fixed size.

    If the text contains fewer tokens than the window size,
    TTR is calculated over the complete text.
    """

    tokens = tokenize(text)

    token_count = len(tokens)

    if token_count == 0:
        return 0.0

    # --------------------------------------------------------
    # Shorter than window
    # --------------------------------------------------------

    if token_count < window_size:

        unique_tokens = len(set(tokens))

        return unique_tokens / token_count

    # --------------------------------------------------------
    # Moving windows
    # --------------------------------------------------------

    ttr_values = []

    for start in range(
        0,
        token_count - window_size + 1
    ):

        window = tokens[
            start:start + window_size
        ]

        unique_tokens = len(set(window))

        ttr = unique_tokens / window_size

        ttr_values.append(ttr)

    return sum(ttr_values) / len(ttr_values)


# ============================================================
# FILE LOADING
# ============================================================

def load_text(file_path):
    """
    Load a UTF-8 text file.
    """

    return file_path.read_text(
        encoding="utf-8"
    ).strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MATTR EVALUATION")
    print("=" * 70)

    print(f"Window size: {WINDOW_SIZE}")
    print(f"Reference directory: {REFERENCE_DIR}")
    print(f"Generated directory: {GENERATED_DIR}")

    results = []

    # ========================================================
    # ORIGINAL + GENERATED
    # ========================================================

    for constraint in CONSTRAINTS:

        print("\n" + "-" * 70)
        print(f"CONSTRAINT: {constraint}")
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
            # Calculate MATTR
            # ------------------------------------------------

            original_mattr = calculate_mattr(
                reference_text
            )

            generated_mattr = calculate_mattr(
                generated_text
            )

            difference = (
                generated_mattr -
                original_mattr
            )

            # Relative change
            if original_mattr != 0:

                relative_change = (
                    difference /
                    original_mattr
                ) * 100

            else:

                relative_change = 0.0

            print(
                f"\n{passage.capitalize()}"
            )

            print(
                f"  Original MATTR : "
                f"{original_mattr:.4f}"
            )

            print(
                f"  Generated MATTR: "
                f"{generated_mattr:.4f}"
            )

            print(
                f"  Difference     : "
                f"{difference:+.4f}"
            )

            print(
                f"  Relative change: "
                f"{relative_change:+.2f}%"
            )

            results.append({

                "Constraint": constraint,

                "Passage": passage,

                "Window_Size": WINDOW_SIZE,

                "Original_MATTR": original_mattr,

                "Generated_MATTR": generated_mattr,

                "Difference": difference,

                "Relative_Change_Percent":
                    relative_change,

            })


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        OUTPUT_DIR /
        "mattr_results.csv"
    )

    fieldnames = [
        "Constraint",
        "Passage",
        "Window_Size",
        "Original_MATTR",
        "Generated_MATTR",
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
    print("MATTR EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
