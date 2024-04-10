#!/bin/bash
#SBATCH --job-name=A5SimSearch  #the name of your job

#change to your NAUID
#SBATCH --output=/scratch/taj262/A5SimSearch.out #this is the file for stdout 
#SBATCH --error=/scratch/taj262/A5SimSearch.err #this is the file for stderr

#SBATCH --time=04:00:00        #Job timelimit is 4 hours
#SBATCH --mem=0        #memory requested in MiB
#SBATCH -G 4 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24    
# #SBATCH --reservation=cs453-spr24-res    

module load cuda 
#compute capability
CC=80
N=7490
DIM=135000
E=10000.0
FNAME=/scratch/taj262/cs453hw/bee_dataset_1D_feature_vectors.txt

nvcc -O3 -arch=compute_$CC -code=sm_$CC -g -lcuda -lineinfo -Xcompiler -fopenmp A5_similarity_search_taj262.cu -o A5_similarity_search_taj262
srun ./A5_similarity_search_taj262 $N $DIM $E $FNAME
