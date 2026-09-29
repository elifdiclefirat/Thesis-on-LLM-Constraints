from pathlib import Path
import re
import csv

PROJECT_ROOT = Path("/netscratch/efirat/thesis")
REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results" / "on"
OUTPUT_DIR = PROJECT_ROOT / "results" / "mattr"
OUTPUT_FILE = OUTPUT_DIR / "on_mattr_results.csv"

CONSTRAINT = "on"
PASSAGES = ["short", "medium", "long"]
WINDOW_SIZE = 100


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text):
    return re.findall(r"\b[\w’'-]+\b", text.lower())


def calculate_mattr(tokens, window_size=100):
    if len(tokens) == 0:
        return 0.0

    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens)

    ttr_values = []

    for start in range(0, len(tokens) - window_size + 1):
        window = tokens[start:start + window_size]
        types = len(set(window))
        tokens_in_window = len(window)
        ttr = types / tokens_in_window
        ttr_values.append(ttr)

    if not ttr_values:
        return 0.0

    return sum(ttr_values) / len(ttr_values)


def calculate_difference(original, generated):
    return generated - original


def calculate_relative_change(original, generated):
    if original == 0:
        return 0.0

    return ((generated - original) / original * 100)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 80)
    print("MATTR Evaluation")
    print("=" * 80)
    print(f"Constraint: {CONSTRAINT.upper()}")
    print(f"Window size: {WINDOW_SIZE}")
    print()

    for passage in PASSAGES:

        reference_file = REFERENCE_DIR / f"passage_{passage}.txt"
        generated_file = GENERATED_DIR / f"passage_{passage}.txt"

        if not reference_file.exists():
            print(f"[WARNING] Reference file not found: {reference_file}")
            continue

        if not generated_file.exists():
            print(f"[WARNING] Generated file not found: {generated_file}")
            continue

        original_text = load_text(reference_file)
        generated_text = load_text(generated_file)

        original_tokens = tokenize(original_text)
        generated_tokens = tokenize(generated_text)

        original_mattr = calculate_mattr(
            original_tokens,
            WINDOW_SIZE
        )

        generated_mattr = calculate_mattr(
            generated_tokens,
            WINDOW_SIZE
        )

        difference = calculate_difference(
            original_mattr,
            generated_mattr
        )

        relative_change = calculate_relative_change(
            original_mattr,
            generated_mattr
        )

        rows.append({
            "Constraint": CONSTRAINT,
            "Passage": passage,
            "Window_Size": WINDOW_SIZE,
            "Original_MATTR": original_mattr,
            "Generated_MATTR": generated_mattr,
            "Difference": difference,
            "Relative_Change_Percent": relative_change,
        })

        print(f"Passage: {passage}")
        print(f"  Original MATTR:  {original_mattr:.6f}")
        print(f"  Generated MATTR: {generated_mattr:.6f}")
        print(f"  Difference:      {difference:.6f}")
        print(f"  Relative change: {relative_change:.2f}%")
        print()

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
