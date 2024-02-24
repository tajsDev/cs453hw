#Compile for the compute capability of the GPU that you are using.
#Compute capability: K80: 37, P100: 60, V100: 70, A100: 80
#See compute capability here for other architectures: https://en.wikipedia.org/wiki/CUDA
#Example compilation for A100

all:
	nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -lineinfo -Xcompiler -fopenmp CS453_A2_taj262.cu -o CS453_A2_taj262
