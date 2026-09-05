from pathlib import Path
import csv

from bert_score import score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "results/bertscore"

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

MODEL_TYPE = "roberta-large"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BERTSCORE EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL_TYPE}")
    print(f"Reference directory: {REFERENCE_DIR}")
    print(f"Generated directory: {GENERATED_DIR}")

    results = []

    for constraint in CONSTRAINTS:

        print("\n" + "-" * 70)
        print(f"CONSTRAINT: {constraint}")
        print("-" * 70)

        for passage in PASSAGES:

            reference_file = (
                REFERENCE_DIR /
                f"passage_{passage}.txt"
            )

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
                    f"WARNING: Missing generated output: "
                    f"{generated_file}"
                )
                continue

            reference = reference_file.read_text(
                encoding="utf-8"
            ).strip()

            candidate = generated_file.read_text(
                encoding="utf-8"
            ).strip()

            print(
                f"\nEvaluating {constraint} / {passage}..."
            )

            precision, recall, f1 = score(
                [candidate],
                [reference],
                model_type=MODEL_TYPE,
                lang="en",
                verbose=False,
            )

            precision_value = precision.item()
            recall_value = recall.item()
            f1_value = f1.item()

            print(
                f"  Precision: {precision_value:.4f}"
            )

            print(
                f"  Recall:    {recall_value:.4f}"
            )

            print(
                f"  F1:        {f1_value:.4f}"
            )

            results.append({
                "Constraint": constraint,
                "Passage": passage,
                "Model": MODEL_TYPE,
                "BERTScore_Precision": precision_value,
                "BERTScore_Recall": recall_value,
                "BERTScore_F1": f1_value,
            })


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        OUTPUT_DIR /
        "bertscore_results.csv"
    )

    fieldnames = [
        "Constraint",
        "Passage",
        "Model",
        "BERTScore_Precision",
        "BERTScore_Recall",
        "BERTScore_F1",
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
    print("BERTSCORE EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Results saved to:\n{output_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
