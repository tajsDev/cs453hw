#!/bin/bash
#SBATCH --job-name=CS453_A0_CUDA_Vector_Add  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/mg2745/A0_CUDA_vector_add_part2.out #this is the file for stdout 
#SBATCH --error=/scratch/mg2745/A0_CUDA_vector_add_part2.err  #this is the file for stderr
														

#SBATCH --time=00:03:00		#Job timelimit is 3 minutes
#SBATCH --mem=20000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C k80 #GPU Model: k80, p100, v100, a100
#SBATCH --reservation=cs453-spr24-res #only the K80 can be used with the reservation

#SBATCH --account=cs453-spr24	#The account is only required if 
								#you already have a Monsoon account
								#and don't want to "charge" your PI 
								#fairshare for your coursework.


#load cuda module
module load cuda/11.7 #for compiling for K80 GPUs
# module load cuda #load default cuda module (P100, V100, A100)

#compile your program (if already compiled then there is no need to compile again here)
#compile for the compute capability of the GPU that you are using. 
#Compute capability: K80: 37, P100: 60, V100: 70, A100: 80
#See compute capability here for other architectures: https://en.wikipedia.org/wiki/CUDA

#compile your program
nvcc -arch=compute_37 -code=sm_37 -lcuda -Xcompiler -fopenmp vector_add_A0.cu -o vector_add_A0_part2

#run your program
srun ./vector_add_A0_part2

