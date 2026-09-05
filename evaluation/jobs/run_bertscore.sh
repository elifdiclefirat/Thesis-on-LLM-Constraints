#!/bin/bash

#SBATCH --job-name=bertscore
#SBATCH --partition=RTXA6000-SLT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00

#SBATCH --output=/netscratch/efirat/thesis/logs/%x-%j.out
#SBATCH --error=/netscratch/efirat/thesis/logs/%x-%j.err


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT=/netscratch/efirat/thesis

source $PROJECT/.venv/bin/activate


# ============================================================
# HUGGING FACE CACHE
# ============================================================

export HF_HOME=$PROJECT/hf-cache
export HF_HUB_CACHE=$PROJECT/hf-cache/hub

export TRANSFORMERS_CACHE=$PROJECT/hf-cache/hub


# ============================================================
# GPU INFORMATION
# ============================================================

echo "============================================================"
echo "BERTSCORE JOB"
echo "============================================================"

echo "Host:"
hostname

echo ""
echo "Python:"
which python

echo ""
echo "PyTorch:"
python -c "import torch; print(torch.__version__)"

echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "GPU:"
nvidia-smi

echo ""
echo "Hugging Face cache:"
echo $HF_HOME

echo "============================================================"


# ============================================================
# RUN BERTSCORE
# ============================================================

cd $PROJECT

python src/evaluate_bertscore.py


# ============================================================
# FINISHED
# ============================================================

echo ""
echo "============================================================"
echo "BERTSCORE JOB FINISHED"
echo "============================================================"
