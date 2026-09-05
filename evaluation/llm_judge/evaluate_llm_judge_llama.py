from pathlib import Path
import csv
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/netscratch/efirat/thesis")

REFERENCE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

GENERATED_DIR = (
    BASE_DIR
    / "results"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llm_judge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "llama3_70b_judge_results_v2.csv"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    "/ds/models/llms/cache/"
    "models--casperhansen--llama-3-70b-instruct-awq/"
    "snapshots/e578178ea893ca5e3326afd15da5aefa37e84d69"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

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


# Llama 3 70B has an 8192-token context window.
MAX_CONTEXT_TOKENS = 8192

# Reserve space for the judge response.
MAX_OUTPUT_TOKENS = 128

# Process a few evaluations at a time.
BATCH_SIZE = 3


# ============================================================
# JUDGE PROMPT
# ============================================================

def build_prompt(reference, generated):
    """
    Create the reference-based evaluation prompt.

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

Do not write "EVALUATION:".
Do not add any text before or after the JSON.

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
# TOKEN-LIMIT HANDLING
# ============================================================

def fit_prompt_to_context(
    reference,
    generated,
    tokenizer,
):
    """
    Make sure the evaluation prompt fits within Llama 3's
    8192-token context window.

    If necessary, truncate the reference and generated passages
    proportionally while preserving their beginning.
    """

    prompt = build_prompt(
        reference,
        generated,
    )

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
    )

    available_tokens = (
        MAX_CONTEXT_TOKENS
        - MAX_OUTPUT_TOKENS
    )

    if len(token_ids) <= available_tokens:
        return prompt

    print(
        f"WARNING: Prompt contains {len(token_ids)} tokens. "
        f"Reducing passage length to fit the context window."
    )

    # --------------------------------------------------------
    # Tokenize passages separately
    # --------------------------------------------------------

    reference_ids = tokenizer.encode(
        reference,
        add_special_tokens=False,
    )

    generated_ids = tokenizer.encode(
        generated,
        add_special_tokens=False,
    )

    # --------------------------------------------------------
    # Calculate template size
    # --------------------------------------------------------

    template_prompt = build_prompt(
        "",
        "",
    )

    template_ids = tokenizer.encode(
        template_prompt,
        add_special_tokens=True,
    )

    template_tokens = len(
        template_ids
    )

    # Leave a safety margin below the hard limit.
    safety_margin = 100

    available_passage_tokens = (
        available_tokens
        - template_tokens
        - safety_margin
    )

    if available_passage_tokens < 1000:
        raise ValueError(
            "Not enough context space for reference and "
            "generated passages."
        )

    # --------------------------------------------------------
    # Split passage budget approximately equally
    # --------------------------------------------------------

    reference_budget = (
        available_passage_tokens // 2
    )

    generated_budget = (
        available_passage_tokens
        - reference_budget
    )

    reference_ids = (
        reference_ids[:reference_budget]
    )

    generated_ids = (
        generated_ids[:generated_budget]
    )

    # --------------------------------------------------------
    # Decode truncated passages
    # --------------------------------------------------------

    reference = tokenizer.decode(
        reference_ids,
        skip_special_tokens=True,
    )

    generated = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------
    # Rebuild prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        reference,
        generated,
    )

    final_token_count = len(
        tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )
    )

    print(
        f"Prompt reduced to {final_token_count} tokens."
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if final_token_count > available_tokens:

        # If the prompt is still too large, reduce both
        # passages further by 5%.
        reference_ids = tokenizer.encode(
            reference,
            add_special_tokens=False,
        )

        generated_ids = tokenizer.encode(
            generated,
            add_special_tokens=False,
        )

        reduced_budget = int(
            available_passage_tokens * 0.95
        )

        reference_budget = (
            reduced_budget // 2
        )

        generated_budget = (
            reduced_budget
            - reference_budget
        )

        reference = tokenizer.decode(
            reference_ids[:reference_budget],
            skip_special_tokens=True,
        )

        generated = tokenizer.decode(
            generated_ids[:generated_budget],
            skip_special_tokens=True,
        )

        prompt = build_prompt(
            reference,
            generated,
        )

        final_token_count = len(
            tokenizer.encode(
                prompt,
                add_special_tokens=True,
            )
        )

        print(
            f"Prompt further reduced to "
            f"{final_token_count} tokens."
        )

    if final_token_count > available_tokens:
        raise ValueError(
            f"Prompt still exceeds context limit: "
            f"{final_token_count} tokens."
        )

    return prompt


# ============================================================
# RESPONSE PARSER
# ============================================================

def parse_judge_response(response):
    """
    Parse the judge response.

    Primary format:
        JSON

    Fallback format:
        Narrative Coherence: 4
        Overall Text Quality: 3
        Justification: ...
    """

    response = response.strip()

    # --------------------------------------------------------
    # 1. Try JSON first
    # --------------------------------------------------------

    try:
        parsed = json.loads(response)

        coherence = parsed.get(
            "coherence"
        )

        overall_quality = parsed.get(
            "overall_quality"
        )

        justification = parsed.get(
            "justification",
            "",
        )

        if (
            isinstance(coherence, int)
            and 1 <= coherence <= 5
            and isinstance(overall_quality, int)
            and 1 <= overall_quality <= 5
        ):
            return {
                "coherence": coherence,
                "overall_quality": overall_quality,
                "justification": str(
                    justification
                ).strip(),
            }

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Try to extract JSON object embedded in response
    # --------------------------------------------------------

    json_match = re.search(
        r"\{.*\}",
        response,
        flags=re.DOTALL,
    )

    if json_match:

        try:
            parsed = json.loads(
                json_match.group(0)
            )

            coherence = parsed.get(
                "coherence"
            )

            overall_quality = parsed.get(
                "overall_quality"
            )

            justification = parsed.get(
                "justification",
                "",
            )

            if (
                isinstance(coherence, int)
                and 1 <= coherence <= 5
                and isinstance(overall_quality, int)
                and 1 <= overall_quality <= 5
            ):
                return {
                    "coherence": coherence,
                    "overall_quality": overall_quality,
                    "justification": str(
                        justification
                    ).strip(),
                }

        except Exception:
            pass

    # --------------------------------------------------------
    # 3. Fallback: natural-language evaluation
    # --------------------------------------------------------

    coherence_match = re.search(
        r"Narrative\s+Coherence\s*:\s*([1-5])",
        response,
        flags=re.IGNORECASE,
    )

    quality_match = re.search(
        r"Overall\s+Text\s+Quality\s*:\s*([1-5])",
        response,
        flags=re.IGNORECASE,
    )

    if not quality_match:

        quality_match = re.search(
            r"Overall[_\s]+Quality\s*:\s*([1-5])",
            response,
            flags=re.IGNORECASE,
        )

    if (
        coherence_match
        and quality_match
    ):

        coherence = int(
            coherence_match.group(1)
        )

        overall_quality = int(
            quality_match.group(1)
        )

        # Extract justification.
        justification = ""

        justification_match = re.search(
            r"Justification\s*:\s*(.*)",
            response,
            flags=re.IGNORECASE
            | re.DOTALL,
        )

        if justification_match:
            justification = (
                justification_match
                .group(1)
                .strip()
            )

        return {
            "coherence": coherence,
            "overall_quality": overall_quality,
            "justification": justification,
        }

    # --------------------------------------------------------
    # 4. Parsing failed
    # --------------------------------------------------------

    print(
        "WARNING: Could not parse judge response."
    )

    return {
        "coherence": "",
        "overall_quality": "",
        "justification": response,
    }


# ============================================================
# CSV SAVING
# ============================================================

def save_results(results):
    """
    Save results to CSV.

    Newlines in Raw_Response and Justification are replaced
    with spaces so that each evaluation occupies one CSV row.
    """

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
        encoding="utf-8",
        newline="",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            clean_result = dict(
                result
            )

            clean_result[
                "Justification"
            ] = (
                str(
                    clean_result.get(
                        "Justification",
                        "",
                    )
                )
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )

            clean_result[
                "Raw_Response"
            ] = (
                str(
                    clean_result.get(
                        "Raw_Response",
                        "",
                    )
                )
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )

            writer.writerow(
                clean_result
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Llama 3 70B LLM-as-a-Judge")
    print("=" * 70)

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    # --------------------------------------------------------
    # Build evaluation prompts
    # --------------------------------------------------------

    prompts = []
    metadata = []

    for constraint in CONSTRAINTS:

        print(
            f"\nPreparing constraint: {constraint}"
        )

        for passage in PASSAGES:

            reference_file = (
                REFERENCE_DIR
                / f"passage_{passage}.txt"
            )

            generated_file = (
                GENERATED_DIR
                / constraint
                / f"passage_{passage}.txt"
            )

            print(
                f"  Checking {constraint}/{passage}"
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

            reference = (
                reference_file.read_text(
                    encoding="utf-8"
                )
            )

            generated = (
                generated_file.read_text(
                    encoding="utf-8"
                )
            )

            try:

                prompt = fit_prompt_to_context(
                    reference,
                    generated,
                    tokenizer,
                )

            except Exception as error:

                print(
                    f"WARNING: Could not prepare "
                    f"{constraint}/{passage}: "
                    f"{error}"
                )

                continue

            prompts.append(
                prompt
            )

            metadata.append({
                "Constraint": constraint,
                "Passage": passage,
            })

    print(
        f"\nTotal evaluations: "
        f"{len(prompts)}"
    )

    expected_evaluations = (
        len(CONSTRAINTS)
        * len(PASSAGES)
    )

    print(
        f"Expected evaluations: "
        f"{expected_evaluations}"
    )

    if len(prompts) != expected_evaluations:

        print(
            "\nWARNING: The number of prepared "
            "evaluations is not 18."
        )

    # --------------------------------------------------------
    # Run judge in batches
    # --------------------------------------------------------

    results = []

    for batch_start in range(
        0,
        len(prompts),
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(prompts),
        )

        batch_prompts = prompts[
            batch_start:batch_end
        ]

        batch_metadata = metadata[
            batch_start:batch_end
        ]

        print("\n" + "=" * 70)

        print(
            f"Processing evaluations "
            f"{batch_start + 1}-{batch_end} "
            f"of {len(prompts)}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        outputs = llm.generate(
            batch_prompts,
            sampling_params,
        )

        batch_results = []

        # ----------------------------------------------------
        # Parse outputs
        # ----------------------------------------------------

        for item, output in zip(
            batch_metadata,
            outputs,
        ):

            response = (
                output.outputs[0]
                .text
                .strip()
            )

            parsed = parse_judge_response(
                response
            )

            result = {
                "Constraint": item[
                    "Constraint"
                ],
                "Passage": item[
                    "Passage"
                ],
                "Coherence": parsed[
                    "coherence"
                ],
                "Overall_Quality": parsed[
                    "overall_quality"
                ],
                "Justification": parsed[
                    "justification"
                ],
                "Raw_Response": response,
            }

            batch_results.append(
                result
            )

        # ----------------------------------------------------
        # Add batch to complete results
        # ----------------------------------------------------

        results.extend(
            batch_results
        )

        # ----------------------------------------------------
        # Save after every batch
        # ----------------------------------------------------

        save_results(
            results
        )

        print(
            f"Saved {len(results)} / "
            f"{len(prompts)} evaluations."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Evaluation completed.")
    print("=" * 70)

    print(
        f"Total results: {len(results)}"
    )

    print(
        f"Results saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
