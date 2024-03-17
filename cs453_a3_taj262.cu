//1-D convolution on the GPU

//Summary of modes 

//              |Global A|Shared A|                    
//              |--------|--------|
//    Global F  |Mode 1  |Mode 5  |
//    Shared F  |Mode 2  |Mode 6  |
// Registers F  |Mode 3  |Mode 7  |
//  Constant F  |Mode 4  |Mode 8  |



#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>


#include "globals.h"

//Note: Normally we don't put CUDA source code into header files. Instead we compile source (*.cu) files separately
//and output object files (*.o) and then link them.
//However, this will make compilation less of a headache by not requiring separate compilation and still allow 
//you to organize your GPU kernels in a separate file instead of having all kernels in this file.
#include "cs453_a3_taj262.cuh"


//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




using namespace std;


//function prototypes
void parameterCheck();
void warmUpGPU();
void compareCPUGPUOutputArrays(float * B_GPU, float * B_CPU, unsigned int NUMELEM);
void printFirstLastArrayElems(float * B, const unsigned int NUMELEM, bool CPUGPU);
void printArrayIfSmall(float * array, const unsigned int NUMELEM, bool CPUGPU);
void computeConvolutionCPU(float * A, float * F, float * B, const unsigned int r, const unsigned int NUMELEM);
void outputSumElems(float * B, unsigned int NUMELEM);


int main(int argc, char *argv[])
{

	
	//Check that R and RNELEMTOTAL are consistent
	//Check that TILESIZE==BLOCKSIZE
	parameterCheck();

	warmUpGPU();
	
	//change OpenMP settings
	//disregard --- only for parallel CPU version if applicable
	omp_set_num_threads(1);

  	//seed random number generator with constant seed
  	//Do not change this or you will be unable to validate your program output
  	srand(123);  

  	//input arrays
  	float * A;
  	float * F;
  	//output arrays
  	float * B;
  	float * B_CPU;

  	A=(float *)malloc(sizeof(float)*N);
  	F=(float *)malloc(sizeof(float)*RNELEMTOTAL);
  	B=(float *)calloc(N,sizeof(float));
  	B_CPU=(float *)calloc(N,sizeof(float));

  	//init input array of FP32 with random numbers between 0 and 1
  	for (unsigned int i=0; i<N; i++){
  		A[i]=(float)rand()/(float)RAND_MAX;
  	}

  	//init filter array of FP32 with random numbers between 0 and 1
  	for (unsigned int i=0; i<RNELEMTOTAL; i++){
  		F[i]=((float)rand()/(float)RAND_MAX);
  	}


	printf("\nMemory requested for 3x arrays of length N (GiB) %f", (3.0*N*sizeof(float)/(1024.0*1024.0*1024.0)));



	///////////////////////////
	//CPU version:
	///////////////////////////

	//Compute the matrix multiplication on the CPU
	//computeConvolutionCPU(A, F, B_CPU, R, N);
	
	//print matrix if N is <=128
	//printArrayIfSmall(B_CPU, N, 0);




	/////////////////////////////
	//GPU
	////////////////////////////	

	
	double tstart=omp_get_wtime();

	float * dev_A;
	float * dev_B;
	float * dev_F;

	//allocate and copy to the device: A, B, and F (if applicable)
	//allocate A on device
	gpuErrchk(cudaMalloc((float**)&dev_A, sizeof(float)*N));	
	//copy A to device
	gpuErrchk(cudaMemcpy(dev_A, A, sizeof(float)*N, cudaMemcpyHostToDevice));
	
	//allocate B on device
	gpuErrchk(cudaMalloc((float**)&dev_B, sizeof(float)*N));	
	//copy B to device (initialized to 0)
	gpuErrchk(cudaMemcpy(dev_B, B, sizeof(float)*N, cudaMemcpyHostToDevice));

	//For these modes we copy the filter array, F, to global memory on the GPU
	if(MODE!=4 && MODE!=8)
	{
	//allocate F on device	
	gpuErrchk(cudaMalloc((float**)&dev_F, sizeof(float)*RNELEMTOTAL));	
	//copy F to device
	gpuErrchk(cudaMemcpy(dev_F, F, sizeof(float)*RNELEMTOTAL, cudaMemcpyHostToDevice));
	}

	//For these modes we copy the filter array, F, to constant memory on the GPU
	if(MODE==4 || MODE==8)
	{
		//allocate F on device	
		
		//copy F to device
		gpuErrchk(cudaMemcpyToSymbol(FGLOBAL,F,sizeof(float)*RNELEMTOTAL));
		//Write code here
	}
	
	
	

	//execute kernels based on MODE
	//MODE==1 refers to storing F in global memory
	if (MODE==1){
		printf("\nMODE==1");
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionGlobalAGlobalF<<<dimGrid, dimBlock>>>(dev_A, dev_F, dev_B, R, N);

	}
	//MODE==2 refers to storing F in shared memory
	else if (MODE==2){
		printf("\nMODE==2");
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionGlobalASharedF<<<dimGrid, dimBlock>>>(dev_A, dev_F, dev_B, R, N);

	}
	//MODE==3 refers to storing a copy of F in each thread's registers
	else if (MODE==3){
		printf("\nMODE==3");
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionGlobalARegistersF<<<dimGrid, dimBlock>>>(dev_A, dev_F, dev_B, R, N);
	}
	//MODE==4 refers to storing F in constant memory
	else if (MODE==4){
		printf("\nMODE==4");
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionGlobalAConstantF<<<dimGrid, dimBlock>>>(dev_A, dev_B, R, N);

	}

	//MODE==5 refers to storing A in shared memory and F in global memory
	else if (MODE==5){
		printf("\nMODE==5");
		//set number of blocks
		unsigned int BLOCKDIM = TILESIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionSharedAGlobalF<<<dimGrid, dimBlock>>>(dev_A, dev_F, dev_B, R, N);

	}

	//MODE==6 refers to storing A in shared memory and F in shared memory
	else if (MODE==6){
		printf("\nMODE==6");
		//set number of blocks
		unsigned int BLOCKDIM = TILESIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionSharedASharedF<<<dimGrid, dimBlock>>>(dev_A, dev_F, dev_B, R, N);

	}
	//MODE==7 refers to storing A in shared memory and F in registers
	else if (MODE==7){
		printf("\nMODE==7");
		//set number of blocks
		unsigned int BLOCKDIM = TILESIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionSharedARegistersF<<<dimGrid, dimBlock>>>(dev_A, dev_F, dev_B, R, N);

	}
	//MODE==8 refers to storing A in shared memory and F in constant memory
	else if (MODE==8){
		printf("\nMODE==8");
		//set number of blocks
		unsigned int BLOCKDIM = TILESIZE; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		convolutionSharedAConstantF<<<dimGrid, dimBlock>>>(dev_A, dev_B, R, N);

	}
	

	//check kernel errors
	gpuErrchk(cudaGetLastError());

	//end execute kernel

	//Copy B from the GPU
	gpuErrchk(cudaMemcpy(B, dev_B, sizeof(float)*N, cudaMemcpyDeviceToHost));
	
	double tend=omp_get_wtime();
	
	printf("\n[MODE: %d, N: %d, R: %d] Total time GPU (s): %f", MODE, N, R, tend-tstart);
	

	//print sum of elements in GPU array B
	outputSumElems(B, N);

	//print matrix if N is <=128
	printArrayIfSmall(B, N, 1);
	
	//Compare CPU and GPU matrices to determine if there are errors in floating point arithmetic or in the GPU code
	compareCPUGPUOutputArrays(B, B_CPU, N);


	//free memory
	free(A);
  	free(F);
  	free(B);
  	free(B_CPU);

  	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_F);

	printf("\n\n");

	return 0;
}

void outputSumElems(float * B, unsigned int NUMELEM)
{
	float sumElems=0;
	for (int i=0; i<NUMELEM; i++)
	{
		sumElems += B[i];
	}
	printf("\n[MODE: %d, N: %d, R: %d] Sum of elems in GPU output matrix (B): %f", MODE, N, R, sumElems);
}

void compareCPUGPUOutputArrays(float * B_GPU, float * B_CPU, unsigned int NUMELEM)
{
	float sumDelta=0;
	float maxDelta=0; //keep track of the maximum difference
	for (int i=0; i<NUMELEM; i++)
	{
		float delta = fabs(B_CPU[i]-B_GPU[i]);
		sumDelta += delta;
		if(maxDelta<delta)
		{
			maxDelta = delta;
		}
	}

	printf("\nSum of deltas between elements: %f",sumDelta);
	printf("\nMaximum delta between elements between output arrays: %f",maxDelta);
}




void warmUpGPU(){
printf("\nWarming up GPU for time trialing...\n");	
cudaDeviceSynchronize();
return;
}



void printArrayIfSmall(float * array, unsigned int NUMELEM, bool CPUGPU)
{
	if(CPUGPU==0)
	{
		printf("\nCPU Array:\n");
	}
	else if(CPUGPU==1)
	{
		printf("\nGPU Array:\n");
	}

	if(NUMELEM<=128)
	{
		unsigned int cnt = 0;
		for (unsigned int i=0; i<NUMELEM; i++)
		{
			cnt++;
			
			if(cnt%10==0)
			{
				printf("%f\n", array[i]);	
			}
			else
			{
				printf("%f, ", array[i]);		
			}

		}
	}
}

void computeConvolutionCPU(float * A, float * F, float * B, const unsigned int r, const unsigned int NUMELEM)
{
	double tstart = omp_get_wtime();
  for(unsigned int i = 0 ; i < NUMELEM ; i ++ ) {
      float localTotal = 0 ;
      for(unsigned int j = 0 ; j <2* r+1 ; j ++ ) {
        unsigned int idx = i+j-r;
        if(idx > 0 && idx < NUMELEM ) {
        localTotal+=A[idx] * F[j];
      }
    }
    B[i] = localTotal;
  }

	double tend = omp_get_wtime();
	printf("\nTotal time CPU (s): %f", tend - tstart);

}


void parameterCheck()
{
	//Ensure BLOCKSIZE==TILESIZE
	if(BLOCKSIZE!=TILESIZE)
	{
		printf("\nError: BLOCKSIZE (%d) and TILESIZE (%d) must be equal\n\n", BLOCKSIZE, TILESIZE);
		fprintf(stderr, "\nError: BLOCKSIZE (%d) and TILESIZE (%d) must be equal\n\n", BLOCKSIZE, TILESIZE);
		exit(0);
	}


	//parameter check
	//Can't have an even value of RNELEMTOTAL
	if(RNELEMTOTAL%2==0)
	{
		printf("\nError: R and RNELEMTOTAL are inconsistent. RNELEMTOTAL!=2R+1 (RNELEMTOTAL cannot be even)\n\n");
		fprintf(stderr, "\nError: R and RNELEMTOTAL are inconsistent. RNELEMTOTAL!=2R+1 (RNELEMTOTAL cannot be even)\n\n");
		exit(0);
	}
	unsigned int tmp1 = (RNELEMTOTAL-1);
	unsigned int tmp2 = tmp1/2;
	if(tmp2!=R)
	{
		printf("\nError: R and RNELEMTOTAL are inconsistent. RNELEMTOTAL!=2R+1\n\n");
		fprintf(stderr, "\nError: R and RNELEMTOTAL are inconsistent. RNELEMTOTAL!=2R+1\n\n");
		exit(0);
	}
}
