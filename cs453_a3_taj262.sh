#!/bin/bash
#SBATCH --job-name=CS453_a3_conv_final  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/taj262/a3.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/a3.err #this is the file for stderr

#SBATCH --time=00:05:00		#Job timelimit is 3 minutes
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24	
#compute capability
module load cuda
CC=70 
#make sure not to redefine MODE, N, etc. in the source file
N=200000000

for M in 1 2 3 4 5 6 7 8
do
	for R in 1500 #5 50 500 #1000 1500 2000 
	do
                RNELEMTOTAL=$((2*R+1))
 		nvcc -O3 -D MODE=$M -D N=$N -D R=$R -D RNELEMTOTAL=$RNELEMTOTAL -diag-suppress 186 -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp cs453_a3_taj262.cu -o cs453_a3_taj262

		#3 time trials
		for i in 1 #2 3
		do
			echo "Mode: $M, R: $R, Trial: $i"
			srun ./cs453_a3_taj262
		done		
	done
done

