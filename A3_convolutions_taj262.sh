#!/bin/bash
#SBATCH --job-name=a3_testcheck  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/taj262/a3_testcheck.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/a3_testcheck.err #this is the file for stderr

#SBATCH --time=00:05:00		#Job timelimit is 3 minutes
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24	
#compute capability
module load cuda
CC=70 
N=200000000
 		nvcc -O3 -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp A3_convolutions_taj262.cu -o A3_convolutions_taj262

		#3 time trials
		for i in 1 #2 3
		do
			srun ./A3_convolutions_taj262
		done		

