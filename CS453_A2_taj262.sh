#!/bin/bash
#SBATCH --job-name=CS453_A2_CUDA #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/taj262/a2.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/a2.err  #this is the file for stderr
														

#SBATCH --time=00:03:00		#Job timelimit is 3 minutes
#SBATCH --mem=20000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100

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
nvcc -O3  -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp -lineinfo CS453_A2_taj262.cu -o CS453_A2_taj262
#run your program
srun ./CS453_A2_taj262 
