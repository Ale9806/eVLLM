#!/bin/bash
set -e

# Datasets array
datasets=(
    'wu_et_al_2023'
    'burgess_et_al_2024_contour'
    'burgess_et_al_2024_eccentricity'
    'burgess_et_al_2024_texture'
    'held_et_al_2010_galt'
    'held_et_al_2010_h2b'
    'held_et_al_2010_mt'
)

# Models array
models=(
    'QwenVLM'
    'PaliGemma'
)

# Iterate over each dataset and model combination
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running inference for dataset: $dataset with model: $model"

        cmd="python src/evlm/inference/model_inference_wrapper.py --dataset_name '$dataset' --model '$model' --do_detection --log_detection_img"
        echo "$cmd"
        eval "$cmd"
    done
done

