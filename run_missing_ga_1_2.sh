#!/bin/bash -l                                                                                                       

#SBATCH --job-name="G_B12"
#SBATCH --time=60:00:00
#SBATCH --partition=compute-p2

#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive
#SBATCH --mem=0

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load miniconda3


srun python run.py --plan structure_experiments/bias_ga/hpc_bias_ga_1_2_s1.yml -mp -r