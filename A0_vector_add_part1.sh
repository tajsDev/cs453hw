#!/bin/bash
#SBATCH --job-name=CS453_A0_CUDA_Vector_Add  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/taj262/A0_CUDA_vector_add_part1.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/A0_CUDA_vector_add_part1.err #this is the file for stderr
														

#SBATCH --time=00:03:00		#Job timelimit is 3 minutes
#SBATCH --mem=20000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C k80 #GPU Model: k80, p100, v100, a100
#SBATCH --reservation=cs453-spr24-res #only the K80 can be used with the reservation

#SBATCH --account=cs453-spr24	#The account is only required if 
								#you already have a Monsoon account
								#and don't want to "charge" your PI 
								#fairshare for your coursework.

#run your program
srun ./vector_add_A0_part1

