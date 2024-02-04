#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>


#define N 1024 
#define MODE 1 //- default 1 thread/element, 2- one thread computes all elements
using namespace std;

void warmUpGPU();
__global__ void addThreadId(unsigned int * A);

int main(int argc, char *argv[])
{
	warmUpGPU();

	unsigned int * A;

	printf("\nSize of A (GiB): %f",(sizeof(unsigned int)*N/(1024.0)));

	A=(unsigned int *)malloc(sizeof(unsigned int)*N);

	//init:
	unsigned int i=0;
	for (i=0; i<N; i++){
		A[i]=i;
	}


	//CPU version:
	double tstartCPU=omp_get_wtime();
	
	for (int i=N-10; i<N; i++)
	{
		printf("\n%d",A[i]);
	}
	double tendCPU=omp_get_wtime();

	double totalTimeCPU = tendCPU-tstartCPU;
	printf("\nTime CPU (s): %0.4f",totalTimeCPU);
	printf("\nThroughput CPU (elems/s): %0.0f", N/totalTimeCPU);


	double tstart=omp_get_wtime();

	
	//CUDA error code:
	cudaError_t errCode=cudaSuccess;
	
	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl; 	
	}

	unsigned int * dev_A;

	//allocate on the device: A, B, C
	errCode=cudaMalloc((unsigned int**)&dev_A, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: A error with code " << errCode << endl; 
	}

	//copy A to device
	errCode=cudaMemcpy( dev_A, A, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl; 
	}	
	

	//execute kernel
	//MODE=1 default, one elem per thread
	if (MODE==1)
	{
	const unsigned int totalBlocks=ceil(N*1.0/256.0);
	printf("\nTotal blocks: %u",totalBlocks);
	addThreadId<<<totalBlocks,256>>>(dev_A);
	}
	else if (MODE==2)
	{
	const unsigned int totalBlocks=1;
	printf("\nTotal blocks: %u",totalBlocks);
	}
	else
	{
		printf("ERROR: invalid kernel mode (must be 1 or 2)");
		return 0;
	}
	//end execute kernel

	//copy data from device to host 
	errCode=cudaMemcpy( A, dev_A, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting C result form GPU error with code " << errCode << endl; 
	}



	cudaDeviceSynchronize();

	//testing -- print last 10 elements
	for (int i=N-10; i<N; i++)
	{
		printf("\n%d",A[i]);
	}
	
	
	double tend=omp_get_wtime();
	double totalTimeGPU = tend-tstart;
	printf("\nTotal time GPU (s): %0.4f",tend-tstart);
	printf("\nThroughput GPU (elems/s): %0.0f", N/totalTimeGPU);



	printf("\n");

	return 0;
}


__global__ void addThreadId(unsigned int * A) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

if (tid>=N){
	return;
}
A[tid]=tid;

return;
}



void warmUpGPU(){


printf("\nWarming up GPU for time trialing...\n");	
cudaDeviceSynchronize();

return;
}
