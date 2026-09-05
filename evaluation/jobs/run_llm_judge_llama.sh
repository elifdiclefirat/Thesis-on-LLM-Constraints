#!/bin/bash
#SBATCH --job-name=llm-judge-llama
#SBATCH --partition=RTXA6000-SLT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/llm_judge_llama_%j.out
#SBATCH --error=logs/llm_judge_llama_%j.err

set -e

echo "======================================================================"
echo "LLM-AS-A-JUDGE — Llama 3 70B Instruct AWQ"
echo "======================================================================"

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"

# ------------------------------------------------------------
# Project environment
# ------------------------------------------------------------

cd /netscratch/efirat/thesis

source .venv/bin/activate

# ------------------------------------------------------------
# CUDA environment
# ------------------------------------------------------------

export HF_HOME=/netscratch/efirat/thesis/hf-cache
export TRANSFORMERS_CACHE=/netscratch/efirat/thesis/hf-cache
export HF_DATASETS_CACHE=/netscratch/efirat/thesis/hf-cache


export CUDA_HOME=/netscratch/efirat/thesis/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# ------------------------------------------------------------
# Hugging Face cache
# ------------------------------------------------------------

export HF_HOME=/netscratch/efirat/thesis/hf-cache

# ------------------------------------------------------------
# Environment check
# ------------------------------------------------------------

echo ""
echo "Python:"
which python
python --version

echo ""
echo "CUDA:"
which nvcc
nvcc --version

echo ""
echo "GPU:"
nvidia-smi

echo ""
echo "HF_HOME:"
echo $HF_HOME

echo ""
echo "======================================================================"
echo "Starting Llama 3 70B Judge"
echo "======================================================================"

python src/evaluate_llm_judge_llama.py

echo ""
echo "======================================================================"
echo "JOB COMPLETED"
echo "======================================================================"

echo "Finished: $(date)"
