#!/bin/bash
set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)

# Cluster parameters
partition="pasteur"
gpu="gpu:a6000"
model=BioMedCLIP" #"random" #"biomedclip" #openclip

# Declare arrays for different configurations
declare -a DATASETS=('acevedo_et_al_2020' 'eulenberg_et_al_2017_darkfield'  'eulenberg_et_al_2017_epifluorescence' 'icpr2020_pollen' 'nirschl_et_al_2018' 'jung_et_al_2022' 'wong_et_al_2022' 'hussain_et_al_2019' 'colocalization_benchmark' 'kather_et_al_2016' 'tang_et_al_2019' 'eulenberg_et_al_2017_brightfield' 'burgess_et_al_2024_contour' 'nirschl_unpub_fluorescence' 'burgess_et_al_2024_eccentricity' 'burgess_et_al_2024_texture' 'held_et_al_2010')


for dataset in "${DATASETS[@]}"; do
    # Construct the command to run
    cmd="python src/evlm/infernece/clip_inference_wrapper.py --dataset_name ${dataset} --model ${model}"
done

