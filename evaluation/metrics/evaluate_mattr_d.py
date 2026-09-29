from pathlib import Path
import re
import csv


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results" / "d"
OUTPUT_DIR = PROJECT_ROOT / "results" / "mattr"

OUTPUT_FILE = OUTPUT_DIR / "d_mattr_results.csv"

CONSTRAINT = "d"

PASSAGES = [
    "short",
    "medium",
    "long",
]

WINDOW_SIZE = 100


# ============================================================
# Text loading
# ============================================================

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Tokenisation
# ============================================================

def tokenize(text):
    """
    Tokenize text into words using the same word-level
    pattern used in the existing evaluation pipeline.
    """

    return re.findall(
        r"\b[\w’'-]+\b",
        text.lower()
    )


# ============================================================
# MATTR calculation
# ============================================================

def calculate_mattr(tokens, window_size=100):
    """
    Calculate Moving-Average Type-Token Ratio (MATTR).

    For each consecutive window of the specified size,
    calculate the type-token ratio and then average
    the resulting values.
    """

    if len(tokens) == 0:
        return 0.0

    if len(tokens) < window_size:
        return (
            len(set(tokens))
            / len(tokens)
        )

    ttr_values = []

    for start in range(
        0,
        len(tokens) - window_size + 1
    ):

        window = tokens[
            start:start + window_size
        ]

        types = len(set(window))
        tokens_in_window = len(window)

        ttr = (
            types
            / tokens_in_window
        )

        ttr_values.append(ttr)

    if not ttr_values:
        return 0.0

    return sum(ttr_values) / len(ttr_values)


# ============================================================
# Difference
# ============================================================

def calculate_difference(
    original,
    generated
):
    return generated - original


# ============================================================
# Relative change
# ============================================================

def calculate_relative_change(
    original,
    generated
):

    if original == 0:
        return 0.0

    return (
        (generated - original)
        / original
        * 100
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    print("=" * 80)
    print("MATTR Evaluation")
    print("=" * 80)
    print(f"Constraint: {CONSTRAINT.upper()}")
    print(f"Window size: {WINDOW_SIZE}")
    print()

    for passage in PASSAGES:

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

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
        # Tokenize
        # ----------------------------------------------------

        original_tokens = tokenize(
            original_text
        )

        generated_tokens = tokenize(
            generated_text
        )

        # ----------------------------------------------------
        # Calculate MATTR
        # ----------------------------------------------------

        original_mattr = calculate_mattr(
            original_tokens,
            WINDOW_SIZE
        )

        generated_mattr = calculate_mattr(
            generated_tokens,
            WINDOW_SIZE
        )

        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        difference = calculate_difference(
            original_mattr,
            generated_mattr
        )

        # ----------------------------------------------------
        # Relative change
        # ----------------------------------------------------

        relative_change = calculate_relative_change(
            original_mattr,
            generated_mattr
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        rows.append({
            "Constraint": CONSTRAINT,
            "Passage": passage,
            "Window_Size": WINDOW_SIZE,
            "Original_MATTR": original_mattr,
            "Generated_MATTR": generated_mattr,
            "Difference": difference,
            "Relative_Change_Percent": relative_change,
        })

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Passage: {passage}"
        )

        print(
            f"  Original MATTR:  "
            f"{original_mattr:.6f}"
        )

        print(
            f"  Generated MATTR: "
            f"{generated_mattr:.6f}"
        )

        print(
            f"  Difference:      "
            f"{difference:.6f}"
        )

        print(
            f"  Relative change: "
            f"{relative_change:.2f}%"
        )

        print()

    # ========================================================
    # Save CSV
    # ========================================================

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
