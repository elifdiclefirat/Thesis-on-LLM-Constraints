from pathlib import Path
import csv

from bert_score import score


PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data" / "processed"
GENERATED_DIR = PROJECT_ROOT / "results" / "on"
OUTPUT_DIR = PROJECT_ROOT / "results" / "bertscore"

OUTPUT_FILE = OUTPUT_DIR / "on_bertscore_results.csv"

CONSTRAINT = "on"
PASSAGES = ["short", "medium", "long"]

MODEL_TYPE = "roberta-large"
LANGUAGE = "en"


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 80)
    print("BERTScore Evaluation")
    print("=" * 80)
    print(f"Constraint: {CONSTRAINT.upper()}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Language: {LANGUAGE}")
    print("Rescaling with baseline: False")
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

        reference = load_text(reference_file)
        candidate = load_text(generated_file)

        precision, recall, f1 = score(
            [candidate],
            [reference],
            model_type=MODEL_TYPE,
            lang=LANGUAGE,
            verbose=False,
            rescale_with_baseline=False,
        )

        precision_value = precision.item()
        recall_value = recall.item()
        f1_value = f1.item()

        rows.append({
            "Constraint": CONSTRAINT,
            "Passage": passage,
            "Model": MODEL_TYPE,
            "Precision": precision_value,
            "Recall": recall_value,
            "F1": f1_value,
        })

        print(f"Passage: {passage}")
        print(f"  Precision: {precision_value:.6f}")
        print(f"  Recall:    {recall_value:.6f}")
        print(f"  F1:        {f1_value:.6f}")
        print()

    fieldnames = [
        "Constraint",
        "Passage",
        "Model",
        "Precision",
        "Recall",
        "F1",
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
