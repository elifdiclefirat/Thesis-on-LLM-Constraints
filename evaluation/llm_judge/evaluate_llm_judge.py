from pathlib import Path
import json
import csv
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = PROJECT_ROOT / "data/processed"
GENERATED_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = PROJECT_ROOT / "results/llm_judge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "qwen3_32b_judge_results.csv"

MODEL_PATH = "/netscratch/efirat/thesis/hf-cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

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
# JUDGE PROMPT
# ============================================================

def build_prompt(reference, generated):
    """
    Create a reference-based evaluation prompt.

    The judge is not told which lexical constraint was used.
    """

    return f"""
You are evaluating the quality of a generated literary passage.

You are given:

1. ORIGINAL PASSAGE
2. GENERATED PASSAGE

Evaluate the generated passage against the original passage.

Do NOT evaluate lexical constraint compliance.
Do NOT assume that the generated passage is good or bad because
of how similar its wording is to the original.

Evaluate ONLY these two dimensions:

1. Narrative Coherence (1-5)
   - logical consistency
   - continuity of events and ideas
   - clarity of the narrative
   - consistency between sentences and paragraphs

2. Overall Text Quality (1-5)
   - fluency
   - meaningfulness
   - linguistic quality
   - overall quality as a literary passage

Score both dimensions independently.

Use the following scale:

1 = Very poor
2 = Poor
3 = Moderate
4 = Good
5 = Excellent

Return ONLY valid JSON in exactly this format:

{{
  "coherence": <integer 1-5>,
  "overall_quality": <integer 1-5>,
  "justification": "<brief explanation>"
}}

ORIGINAL PASSAGE:
<<<
{reference}
>>>

GENERATED PASSAGE:
<<<
{generated}
>>>
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Extract the first JSON object from the model response.
    """

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "No JSON object found in model response."
        )

    return json.loads(match.group(0))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print("Judge: Qwen3-32B")
    print("Dimensions: Coherence + Overall Quality")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
	enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    prompts = []
    metadata = []

    # --------------------------------------------------------
    # Build all evaluation prompts
    # --------------------------------------------------------

    for constraint in CONSTRAINTS:

        print(
            f"\nPreparing constraint: {constraint}"
        )

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
                    f"WARNING: Missing generated: "
                    f"{generated_file}"
                )
                continue

            reference = reference_file.read_text(
                encoding="utf-8"
            )

            generated = generated_file.read_text(
                encoding="utf-8"
            )

            prompt = build_prompt(
                reference,
                generated
            )

            prompts.append(prompt)

            metadata.append({
                "Constraint": constraint,
                "Passage": passage,
            })

    print(
        f"\nTotal evaluations: {len(prompts)}"
    )

    # --------------------------------------------------------
    # Run judge
    # --------------------------------------------------------

    outputs = llm.generate(
        prompts,
        sampling_params
    )

    results = []

    # --------------------------------------------------------
    # Parse outputs
    # --------------------------------------------------------

    for meta, output in zip(
        metadata,
        outputs
    ):

        raw_response = (
            output.outputs[0].text.strip()
        )

        print("\n" + "-" * 70)

        print(
            f"Constraint: {meta['Constraint']}"
        )

        print(
            f"Passage: {meta['Passage']}"
        )

        print(
            f"Raw response:\n{raw_response}"
        )

        try:

            evaluation = extract_json(
                raw_response
            )

            coherence = evaluation[
                "coherence"
            ]

            overall_quality = evaluation[
                "overall_quality"
            ]

            justification = evaluation.get(
                "justification",
                ""
            )

        except Exception as error:

            print(
                f"WARNING: Could not parse JSON: "
                f"{error}"
            )

            coherence = None
            overall_quality = None
            justification = raw_response

        results.append({

            "Constraint":
                meta["Constraint"],

            "Passage":
                meta["Passage"],

            "Coherence":
                coherence,

            "Overall_Quality":
                overall_quality,

            "Justification":
                justification,

            "Raw_Response":
                raw_response,
        })

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = [
        "Constraint",
        "Passage",
        "Coherence",
        "Overall_Quality",
        "Justification",
        "Raw_Response",
    ]

    with open(
        OUTPUT_FILE,
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

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LLM-AS-A-JUDGE COMPLETED")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
