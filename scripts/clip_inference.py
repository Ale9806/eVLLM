#!/usr/bin/env python
import os
import subprocess
from datetime import datetime

# Cluster parameters
partition = "pasteur"
account = "pasteur"
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
gpu = "gpu"

# Declare arrays for different configurations
DATASETS = [
    'acevedo_et_al_2020', 'eulenberg_et_al_2017_darkfield',
    'eulenberg_et_al_2017_epifluorescence', 'icpr2020_pollen',
    'nirschl_et_al_2018', 'jung_et_al_2022', 'wong_et_al_2022',
    'hussain_et_al_2019', 'colocalization_benchmark', 'kather_et_al_2016',
    'tang_et_al_2019', 'eulenberg_et_al_2017_brightfield',
    'burgess_et_al_2024_contour', 'nirschl_unpub_fluorescence',
    'burgess_et_al_2024_eccentricity', 'burgess_et_al_2024_texture',
    'held_et_al_2010'
]
model = "BioMedCLIP"  # "random" #"biomedclip" #openclip

for dataset in DATASETS:
    # Construct the command to run
    cmd = f"python src/evlm/infernece/clip_inference_wrapper.py --dataset_name {dataset} --model {model}"
    print("Constructed Command:", cmd)

    # Construct job script content
    job_script = f"""#!/bin/bash
#SBATCH --job-name={dataset}-
#SBATCH --output=slurm_logs/{dataset}-{current_date}-%j-out.txt
#SBATCH --error=slurm_logs/{dataset}-{current_date}-{os.path.basename(__file__)}%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 2
#SBATCH --gres={gpu}
#SBATCH -p {partition}
#SBATCH -A {account}
#SBATCH --time=48:00:00
#SBATCH --ntasks=1 

echo "{cmd}"
# Uncomment below to actually run the command
# eval "{cmd}"
"""

    # Write job script to file
    job_script_path = f"job_{dataset}.sh"
    with open(job_script_path, "w") as f:
        f.write(job_script)

    # Submit the job
    subprocess.run(["sbatch", job_script_path])

    # Remove the job script file
    os.remove(job_script_path)
