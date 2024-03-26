#!/bin/bash
#SBATCH --job-name=CS453_a4_new  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/taj262/a4_new.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/a4_new.err #this is the file for stderr

#SBATCH --time=00:05:00		#Job timelimit is 3 minutes
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#compute capability
module load cuda
CC=70 
#make sure not to redefine MODE, N, etc. in the source file
COARSEFACTOR=16
N=536870912
for M in 1 2 3
do 
		nvcc -O3 -D MODE=$M -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp A4_reductions_taj262.cu -o A4_reductions_taj262

		#3 time trials
		for i in 1 2 3
		do
			echo "Mode: $M,Trial: $i N: $N"
			srun ./A4_reductions_taj262 $N
		done
done
