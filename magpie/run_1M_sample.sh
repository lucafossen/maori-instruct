#!/usr/bin/env bash
#SBATCH --qos=serial
#SBATCH --job-name=SAMPLE_INSTRUCTIONS
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=.slurm/SAMPLE_INSTRUCTIONS.log
#SBATCH --error=.slurm/SAMPLE_INSTRUCTIONS.err
#SBATCH --mail-type=REQUEUE
#SBATCH --time=00:05:00
#SBATCH --mail-user=oscar.sainz@ehu.eus

source /scratch/osainz006/LatxaTxat/venv/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

cd instruct_translation

python sample.py