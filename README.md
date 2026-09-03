# Evaluating the Robustness of Automatic Evaluation Metrics under Progressive Lexical Constraints in Long-Form Text Generation

Master's thesis code repository — Elif Dicle Fırat, GISMA University of Applied Sciences.
Supervisor: Philippe Thomas.

## Overview

This repository contains the code used to generate lipogram-constrained rewrites of
Arthur Conan Doyle's *The Adventures of Sherlock Holmes* using Qwen3-32B, and to
evaluate the robustness of automatic evaluation metrics (BERTScore, MATTR, Flesch
Reading Ease, constraint-compliance checks, LLM-as-a-Judge) as lexical constraint
severity increases.

## Repository structure

```
generation/     Scripts for prompting Qwen3-32B and producing constrained rewrites
evaluation/     Scripts for computing BERTScore, MATTR, Flesch Reading Ease,
                constraint-compliance checks, and LLM-as-a-Judge scoring
analysis/       Notebooks/scripts for aggregating results and producing figures
data/           Source passages and generated outputs (see note below on data)
```

## Data note

Source texts are taken from *The Adventures of Sherlock Holmes* (Doyle, 1892),
available via Project Gutenberg: https://www.gutenberg.org/ebooks/1661

## Model

Generation uses Qwen3-32B (https://huggingface.co/Qwen/Qwen3-32B) run locally via
vLLM, with `enable_thinking=False`.

## Status

Work in progress — this repository is being updated as experiments are completed.
