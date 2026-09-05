#!/bin/bash

#SBATCH --job-name=llm-judge-qwen
#SBATCH --partition=RTXA6000-SLT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/efirat/thesis/logs/llm_judge_qwen_%j.out
#SBATCH --error=/netscratch/efirat/thesis/logs/llm_judge_qwen_%j.err

cd /netscratch/efirat/thesis

source .venv/bin/activate

export HF_HOME=/netscratch/efirat/thesis/hf-cache
export TRANSFORMERS_CACHE=/netscratch/efirat/thesis/hf-cache
export HF_DATASETS_CACHE=/netscratch/efirat/thesis/hf-cache

export CUDA_HOME=/netscratch/efirat/thesis/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "===== CUDA DEBUG ====="
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "which nvcc:"
which nvcc
echo "nvcc:"
nvcc --version
echo "Python:"
which python
python --version
echo "======================"


python src/evaluate_llm_judge.py
