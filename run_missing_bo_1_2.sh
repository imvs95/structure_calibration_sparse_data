#!/bin/bash -l                                                                                                       

#SBATCH --job-name="BO_B12"
#SBATCH --time=100:00:00
#SBATCH --partition=compute-p2

#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive
#SBATCH --mem=0

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load miniconda3


srun python run.py --plan structure_experiments/bias_bo/hpc_bias_bo_1_2_s1.yml -mp -r