#!/bin/bash -l                                                                                                       

#SBATCH --job-name="TestAll"
#SBATCH --time=10:00:00
#SBATCH --partition=compute 

#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive
#SBATCH --mem=0

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load miniconda3


srun python run.py --plan structure_experiments/hpc_test_all_models.yml -mp -r