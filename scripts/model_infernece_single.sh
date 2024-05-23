#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=slurm_logs/output-%x.log
#SBATCH --error=slurm_logs/error-%x.log
#SBATCH -p pasteur
#SBATCH -A pasteur
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=40G

#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model BioMedCLIP

python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model CogVLM
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model QwenVLM





