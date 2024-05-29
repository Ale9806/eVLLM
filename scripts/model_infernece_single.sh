#!/bin/bash
#SBATCH --job-name=test_single
#SBATCH --output=slurm_logs/output-%x.log
#SBATCH --error=slurm_logs/error-%x.log
#SBATCH -p pasteur
#SBATCH -A pasteur
#SBATCH --gres=gpu:a6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=40G

# Contrastive: 
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model all
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model ALIGN
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model BioMedCLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model CLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model OpenCLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model QuiltCLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model PLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model BLIP
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model ConchCLIP

#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model CogVLM
#python src/evlm/inference/model_inference_wrapper.py --dataset_name all --model QwenVLM


# comand: sbatch scripts/model_infernece_single.sh


