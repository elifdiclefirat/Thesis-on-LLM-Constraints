from vllm import LLM, SamplingParams
from pathlib import Path
import argparse


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "Qwen/Qwen3-32B"

CONSTRAINTS = {
    "e": ["e"],
    "et": ["e", "t"],
    "eta": ["e", "t", "a"],
    "etao": ["e", "t", "a", "o"],
    "etaoi": ["e", "t", "a", "o", "i"],
    "etaoin": ["e", "t", "a", "o", "i", "n"],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = PROJECT_ROOT / "results"


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate lexically constrained versions "
                "of the experimental passages."
)

parser.add_argument(
    "--constraint",
    required=True,
    choices=CONSTRAINTS.keys(),
    help="Constraint level to run."
)

args = parser.parse_args()

constraint_name = args.constraint
forbidden_letters = CONSTRAINTS[constraint_name]

output_dir = OUTPUT_ROOT / constraint_name
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# PASSAGES
# ============================================================

passage_names = [
    "passage_short.txt",
    "passage_medium.txt",
    "passage_long.txt",
]

passages = [
    INPUT_DIR / name
    for name in passage_names
]

missing_passages = [
    path
    for path in passages
    if not path.exists()
]

if missing_passages:
    raise FileNotFoundError(
        "The following passage files are missing:\n"
        + "\n".join(str(path) for path in missing_passages)
    )


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

print("=" * 70)
print("LEXICAL CONSTRAINT GENERATION")
print("=" * 70)

print(f"Model:              {MODEL}")
print(f"Constraint:         {constraint_name}")
print(f"Forbidden letters:  {forbidden_letters}")
print(f"Input directory:    {INPUT_DIR}")
print(f"Output directory:   {output_dir}")
print(f"Number of passages: {len(passages)}")

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen3-32B...")

llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    max_model_len=32768,
    enforce_eager=True,
)


# ============================================================
# SAMPLING PARAMETERS
# ============================================================

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0,
    presence_penalty=1.5,
    max_tokens=22000,
)


# ============================================================
# GENERATE PASSAGES
# ============================================================

for passage_file in passages:

    output_file = output_dir / passage_file.name

    # --------------------------------------------------------
    # Do not overwrite existing results
    # --------------------------------------------------------

    if output_file.exists():
        print(
            f"\nSkipping {passage_file.name}: "
            f"output already exists."
        )
        continue

    print("\n" + "=" * 70)
    print(f"PASSAGE: {passage_file.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Read passage
    # --------------------------------------------------------

    with open(
        passage_file,
        "r",
        encoding="utf-8"
    ) as f:
        passage = f.read()

    input_words = len(passage.split())

    forbidden = ", ".join(forbidden_letters)

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""Rewrite the following passage in full.

Preserve all information, events, actions, characters,
relationships, descriptions, and chronological details
from the original passage.

Do not summarize, shorten, omit, or skip any part of
the original passage.

The rewritten passage must cover the complete original
passage from beginning to end.

Apply a strict character-level lipogram.

The following letters are forbidden:
{forbidden}

The forbidden letters must not appear anywhere in the
generated passage, regardless of capitalization.

Use alternative words, grammatical constructions, or
paraphrases when necessary to avoid the forbidden letters,
while preserving the original meaning and information.

Return ONLY the rewritten passage.

Do not explain the task.
Do not provide reasoning.
Do not describe your approach.
Do not mention the constraint.
Do not mention the rewriting process.
Do not add commentary before or after the rewritten passage.

ORIGINAL PASSAGE:

{passage}
"""

    print(f"Input words:        {input_words}")
    print(f"Forbidden letters:  {forbidden}")
    print("Thinking mode:      OFF")
    print("Presence penalty:   1.5")
    print("Generating...")

    # --------------------------------------------------------
    # Qwen3 chat generation
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
        chat_template_kwargs={
            "enable_thinking": False,
        },
    )

    generated_text = outputs[0].outputs[0].text.strip()

    output_words = len(generated_text.split())

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(generated_text)

    print(f"Output words:       {output_words}")
    print(f"Saved to:            {output_file}")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("ALL PASSAGES COMPLETED")
print("=" * 70)

print(f"Constraint:       {constraint_name}")
print(f"Output directory: {output_dir}")

print("=" * 70)
