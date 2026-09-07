#! /bin/bash

SLURM_DIR=$(pwd)"/.slurm"

cd instruct_translation

SPLITS=(
    "0 500000"
    "500000 1000000"
    "1000000 1500000"
    "1500000 2000000"
    "2000000 2500000"
    "2500000 3000000"
)

for i in "${!SPLITS[@]}"; do
    SPLIT=(${SPLITS[$i]})
    echo "Starting translation for split: ${SPLIT}"
    SLURM_ARGS=$(
        echo \
            --partition=general \
            --qos=xlong \
            --job-name="dataset_translation_${i}" \
            --mem=32G \
            --gres=gpu:1 \
            --constraint=a100 \
            --cpus-per-task=16 \
	    --time=8-00:00:00 \
            --output="${SLURM_DIR}/dataset_translation_${SPLIT[0]}_${SPLIT[1]}.out" \
            --error="${SLURM_DIR}/dataset_translation_${SPLIT[0]}_${SPLIT[1]}.err"
    )
    echo "SLURM_ARGS: ${SLURM_ARGS}"
    sbatch ${SLURM_ARGS} run_single_translation.sh \
        $DATA/models/Latxa3.1_8b_lr1e-5 \
        HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered \
        prompt.j2 \
        0.15 \
        1 \
        ${SPLIT[0]} \
        ${SPLIT[1]}
done
