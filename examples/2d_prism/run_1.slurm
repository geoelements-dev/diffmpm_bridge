#!/bin/bash
#SBATCH --job-name=diffmpm
#SBATCH --output=results/o%j.txt
#SBATCH --error=results/e%j.txt
#SBATCH --partition=normal     
#SBATCH --time=48:00:00   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END            
#SBATCH --mail-user=jeffreycheng@berkeley.edu
#SBATCH -A BCS20003

# Activate environment
set -e
source start_venv_frontera.sh

# Define paths
# mkdir "results"

# Run Training
python true_randomized_1.py