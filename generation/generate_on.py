from vllm import LLM, SamplingParams
from pathlib import Path
import argparse


MODEL = "Qwen/Qwen3-32B"

CONSTRAINTS = {
    "on": ["on"],
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = PROJECT_ROOT / "results"


parser = argparse.ArgumentParser(
    description="Generate a word-level constrained version "
                "with an 'on' constraint."
)

parser.add_argument(
    "--constraint",
    required=True,
    choices=CONSTRAINTS.keys(),
    help="Constraint level to run."
)

args = parser.parse_args()

constraint_name = args.constraint
forbidden_words = CONSTRAINTS[constraint_name]

output_dir = OUTPUT_ROOT / constraint_name
output_dir.mkdir(parents=True, exist_ok=True)


passage_names = [
    "passage_short.txt",
    "passage_medium.txt",
    "passage_long.txt",
]

passages = [INPUT_DIR / name for name in passage_names]

missing_passages = [
    path for path in passages
    if not path.exists()
]

if missing_passages:
    raise FileNotFoundError(
        "The following passage files are missing:\n"
        + "\n".join(str(path) for path in missing_passages)
    )


print("=" * 70)
print("LEXICAL CONSTRAINT GENERATION")
print("=" * 70)
print(f"Model:              {MODEL}")
print(f"Constraint:         {constraint_name}")
print(f"Forbidden words:    {forbidden_words}")
print(f"Input directory:    {INPUT_DIR}")
print(f"Output directory:   {output_dir}")
print(f"Number of passages: {len(passages)}")
print("=" * 70)


print("\nLoading Qwen3-32B...")

llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    max_model_len=32768,
    enforce_eager=True,
)


sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0,
    presence_penalty=1.5,
    max_tokens=22000,
)


for passage_file in passages:

    output_file = output_dir / passage_file.name

    if output_file.exists():
        print(
            f"\nSkipping {passage_file.name}: "
            f"output already exists."
        )
        continue

    print("\n" + "=" * 70)
    print(f"PASSAGE: {passage_file.name}")
    print("=" * 70)

    with open(passage_file, "r", encoding="utf-8") as f:
        passage = f.read()

    input_words = len(passage.split())

    forbidden = ", ".join(forbidden_words)

    prompt = f"""Rewrite the following passage in full.

Preserve all information, events, actions, characters,
relationships, descriptions, and chronological details
from the original passage.

Do not summarize, shorten, omit, or skip any part of
the original passage.

The rewritten passage must cover the complete original
passage from beginning to end.

Apply a strict word-level constraint.

The following word is forbidden:
{forbidden}

The forbidden word must not appear anywhere in the
generated passage as an independent word.

Words that contain the sequence "on" as part of a larger
word are allowed, such as "only", "once", "upon", or
"condition".

Use alternative words, grammatical constructions, or
paraphrases when necessary to avoid the forbidden word,
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
    print(f"Forbidden words:    {forbidden_words}")
    print("Constraint type:    word-level")
    print("Thinking mode:      OFF")
    print("Presence penalty:   1.5")
    print("Generating...")

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    generated_text = outputs[0].outputs[0].text.strip()

    output_words = len(generated_text.split())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Output words:       {output_words}")
    print(f"Saved to:            {output_file}")


print("\n" + "=" * 70)
print("ALL PASSAGES COMPLETED")
print("=" * 70)
print(f"Constraint:       {constraint_name}")
print(f"Output directory: {output_dir}")
print("=" * 70)

