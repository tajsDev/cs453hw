#!/bin/bash
#SBATCH --job-name=A4_reductions_taj262  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/taj262/A4_reductions_taj262.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/A4_reductions_taj262.err #this is the file for stderr

#SBATCH --time=00:10:00		#Job timelimit is 3 minutes
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#compute capability
module load cuda
CC=70 
#make sure not to redefine MODE, N, etc. in the source file
COARSE=4
N=536870912
for M in 1 2 3 4 5
do 
   for B in 256 #32 64 128 256 512 1024
   do
		nvcc -O3 -D MODE=$M -D BLOCKSIZE=$B -D COARSEFACTOR=$COARSE -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp A4_reductions_taj262.cu -o A4_reductions_taj262

		#3 time trials
		for i in 1 #2 3
		do
			echo "Mode: $M,Trial: $i N: $N ,Blocksize: $B"
			srun ./A4_reductions_taj262 $N
		done
   done
done
