datasets=("acevedo_et_al_2020"
          "burgess_et_al_2024_contour"
          "burgess_et_al_2024_eccentricity"
          "burgess_et_al_2024_texture"
          "colocalization_benchmark"
          "eulenberg_et_al_2017_brightfield"
          "eulenberg_et_al_2017_darkfield"
          "eulenberg_et_al_2017_epifluorescence"
          "held_et_al_2010_galt"
          "held_et_al_2010_h2b"
          "held_et_al_2010_mt"
          "hussain_et_al_2019"
          "icpr2020_pollen"
          "jung_et_al_2022"
          "kather_et_al_2016"
          "kather_et_al_2018"
          "kather_et_al_2018_val7k"
          "nirschl_et_al_2018"
          "nirschl_unpub_fluorescence"
          "tang_et_al_2019"
          "wong_et_al_2022"
          "wu_et_al_2023"
          "empiar_sbfsem")

# Iterate over each dataset
for dataset in "${datasets[@]}"
do
  # Create a temporary submission script
  temp_script=$(mktemp)
  cat <<EOT > $temp_script
#!/bin/bash
#SBATCH --job-name=test_${dataset}
#SBATCH --output=slurm_logs/output-${dataset}.log
#SBATCH --error=slurm_logs/error-${dataset}.log
#SBATCH -p pasteur
#SBATCH -A pasteur
#SBATCH --gres=gpu:l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=70G

# Contrastive:
python src/evlm/inference/model_inference_wrapper.py --dataset_name $dataset --model QwenVLM
EOT



  # Submit the job
  sbatch $temp_script

  # Optionally, remove the temporary script after submission
  rm $temp_script
done