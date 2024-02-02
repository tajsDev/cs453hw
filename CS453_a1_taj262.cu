#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>


#define N 10000
#define MODE 2 //- default 1 thread/element, 2- one thread computes all elements
using namespace std;

void warmUpGPU();
__global__ void vectorAdd(unsigned int * A, unsigned int * B, unsigned int * C);
__global__ void vectorAddOneThreadAllElems(unsigned int * A, unsigned int * B, unsigned int * C, const unsigned int NELEMS);

int main(int argc, char *argv[])
{
	warmUpGPU();

	unsigned int * A;
	unsigned int * B;
	unsigned int * C;
	unsigned int * C_CPU;

	printf("\nSize of A+B+C (GiB): %f",(sizeof(unsigned int)*N*3.0)/(1024.0*1024.0*1024.0));

	A=(unsigned int *)malloc(sizeof(unsigned int)*N);
	B=(unsigned int *)malloc(sizeof(unsigned int)*N);
	C=(unsigned int *)malloc(sizeof(unsigned int)*N);
	C_CPU=(unsigned int *)malloc(sizeof(unsigned int)*N);


	//init:
	unsigned int i=0;
	for (i=0; i<N; i++){
		A[i]=i;
		B[i]=i;
		C[i]=0;
		C_CPU[i]=0;
	}


	//CPU version:
	double tstartCPU=omp_get_wtime();
	
	for (int i=0; i<N; i++){
		C_CPU[i]=A[i]+B[i];
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
	unsigned int * dev_B;
	unsigned int * dev_C;

	//allocate on the device: A, B, C
	errCode=cudaMalloc((unsigned int**)&dev_A, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: A error with code " << errCode << endl; 
	}

	errCode=cudaMalloc((unsigned int**)&dev_B, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: B error with code " << errCode << endl; 
	}

	errCode=cudaMalloc((unsigned int**)&dev_C, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: C error with code " << errCode << endl; 
	}

	//copy A to device
	errCode=cudaMemcpy( dev_A, A, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl; 
	}	
	
	//copy B to device
	errCode=cudaMemcpy( dev_B, B, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl; 
	}

	//copy C to device (initialized to 0)
	errCode=cudaMemcpy( dev_C, C, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: A memcpy error with code " << errCode << endl; 
	}

	//execute kernel
	//MODE=1 default, one elem per thread
	if (MODE==1)
	{
	const unsigned int totalBlocks=ceil(N*1.0/1024.0);
	printf("\nTotal blocks: %u",totalBlocks);
	vectorAdd<<<totalBlocks,1024>>>(dev_A, dev_B, dev_C);
	}
	else if (MODE==2)
	{
	const unsigned int totalBlocks=1;
	printf("\nTotal blocks: %u",totalBlocks);
	vectorAddOneThreadAllElems<<<totalBlocks,1024>>>(dev_A, dev_B, dev_C, N);	
	}
	else
	{
		printf("ERROR: invalid kernel mode (must be 1 or 2)");
		return 0;
	}
	//end execute kernel

	//copy data from device to host 
	errCode=cudaMemcpy( C, dev_C, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: getting C result form GPU error with code " << errCode << endl; 
	}



	cudaDeviceSynchronize();

	//testing -- print last 10 elements
	for (int i=N-10; i<N; i++)
	{
		printf("\n%d",C[i]);
	}
	
	
	double tend=omp_get_wtime();
	double totalTimeGPU = tend-tstart;
	printf("\nTotal time GPU (s): %0.4f",tend-tstart);
	printf("\nThroughput GPU (elems/s): %0.0f", N/totalTimeGPU);



	printf("\n");

	return 0;
}


__global__ void vectorAdd(unsigned int * A, unsigned int * B, unsigned int * C) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

if (tid>=N){
	return;
}
C[tid]=A[tid]+B[tid];

return;
}

__global__ void vectorAddOneThreadAllElems(unsigned int * A, unsigned int * B, unsigned int * C, const unsigned int NELEMS) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

if (tid>=1){
	return;
}

for (unsigned int i=0; i<NELEMS; i++)
{
	C[i]=A[i]+B[i];
}

return;
}



void warmUpGPU(){


printf("\nWarming up GPU for time trialing...\n");	
cudaDeviceSynchronize();

return;
}