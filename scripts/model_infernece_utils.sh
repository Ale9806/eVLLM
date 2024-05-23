#!/bin/bash
#SBATCH --job-name=$4_$3
#SBATCH --output=test_.log
#SBATCH --error=errpr.log
#SBATCH -p $1
#SBATCH -A $1
#SBATCH --gres=$2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=30G

python src/evlm/inference/model_inference_wrapper.py --dataset_name $3 --model $4







