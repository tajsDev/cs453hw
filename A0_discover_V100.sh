#!/bin/bash
#SBATCH --job-name=CS453_A0_Discover_V100  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/taj262/A0_Discover_V100.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/A0_Discover_V100.err #this is the file for stderr

#SBATCH --time=00:03:00		#Job timelimit is 3 minutes
#SBATCH --mem=1000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24	
								
								
srun hostname
srun nvidia-smi
