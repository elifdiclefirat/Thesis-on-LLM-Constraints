#!/bin/bash
#SBATCH --job-name=lipogram_on
#SBATCH --partition=RTXA6000-SLT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lipogram_on_%j.out
#SBATCH --error=logs/lipogram_on_%j.err

set -e

echo "======================================================================"
echo "LIPOGRAM EXPERIMENT"
echo "======================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Constraint:   word-level 'on'"
echo "Started:      $(date)"
echo "======================================================================"

export CUDA_HOME=/netscratch/efirat/thesis/test-py311/lib/python3.11/site-packages/nvidia/cu13
export PATH=/netscratch/efirat/thesis/test-py311/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export HF_HOME=/netscratch/efirat/thesis/hf-cache
export HF_HUB_CACHE=/netscratch/efirat/thesis/hf-cache/hub
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

echo ""
echo "Environment:"
echo "CUDA_HOME:       $CUDA_HOME"
echo "nvcc:             $(which nvcc)"
echo "ninja:            $(which ninja)"
echo "ninja version:    $(ninja --version)"
echo "LD_LIBRARY_PATH:  $LD_LIBRARY_PATH"
echo "HF_HOME:          $HF_HOME"
echo "HF_HUB_CACHE:     $HF_HUB_CACHE"
echo "VLLM_ENABLE_V1_MULTIPROCESSING: $VLLM_ENABLE_V1_MULTIPROCESSING"
echo "VLLM_ALLREDUCE_USE_FLASHINFER:   $VLLM_ALLREDUCE_USE_FLASHINFER"
echo "VLLM_USE_FLASHINFER_SAMPLER:     $VLLM_USE_FLASHINFER_SAMPLER"

PYTHON="/netscratch/efirat/thesis/test-py311/bin/python"

echo ""
echo "Python: $PYTHON"
"$PYTHON" --version

echo ""
echo "GPU information:"
nvidia-smi

echo ""
echo "Starting generation..."
"$PYTHON" src/generate_on.py --constraint on

echo ""
echo "======================================================================"
echo "GENERATION COMPLETED"
echo "Finished: $(date)"
echo "======================================================================"
