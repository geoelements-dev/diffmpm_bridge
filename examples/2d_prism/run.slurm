#!/bin/bash
#SBATCH --job-name=diffmpm_test
#SBATCH --output=results/diffmpm_test.o%j
#SBATCH --error=results/diffmpm_test.e%j
#SBATCH --partition=rtx     
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
mkdir "resuts"

# Run Training
python true_randomized.py