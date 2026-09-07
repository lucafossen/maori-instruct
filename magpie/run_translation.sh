#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --job-name=dataset_translation_2500000_3000000
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=32G
#SBATCH --output=.slurm/dataset_translation_2500000_3000000.out
#SBATCH --error=.slurm/dataset_translation_2500000_3000000.err

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

python translate.py \
    --model_path $DATA/models/Latxa3.1_8b_lr1e-5 \
    --dataset_path HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered \
    --prompt_path prompt.j2 \
    --frequency_penalty 0.15 \
    --tensor_parallel_size 1 \
    --dataset_start 2500000 \
    --dataset_end 3000000
