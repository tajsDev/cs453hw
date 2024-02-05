//Note, when compiling you will likely observe numerous warnings
//because I have commented out a bunch of the code in the starter file

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>


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


// You may prefer to move these parameters to your job script
#define N 100 //Elements in N
#define MODE 1 //GPU Mode
#define NUMELEMPERTHREAD 8 //This is r in the assignment instructions for MODE 3
#define BLOCKSIZE 1024 //GPU CUDA block size for the first two kernels for MODES 2-4.


using namespace std;


//function prototypes

//CPU function that computes the standard deviation
double computeStdDevCPU(double * A, const unsigned int NUMELEM);

//GPU kernels:
//Mode 1:
__global__ void computeMeanOneThread(double *A, double *mean, const unsigned int NUMELEM);
__global__ void computeStdDevOneThread(double *A, double *mean, const unsigned int NUMELEM, double *stddev);


//Mode 2:
__global__ void computeGlobalSumWithOneElemPerThread(double *A, const unsigned int NUMELEM, double *globalSum);
__global__ void computeSumDiffElemWithOneElemPerThread(double *A, const unsigned int NUMELEM, double *globalSum, double *globalSumDiffElemTotalSumSquared);


//Mode 3:
__global__ void computeGlobalSumWithMultipleElemsPerThread(double *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD, double *globalSum);
__global__ void computeSumDiffElemWithMultipleElemsPerThread(double *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD,  double *globalSum, double *globalSumDiffElemTotalSumSquared);


//Mode 4:
__global__ void computeGlobalSumWithOneElemPerThreadWithSM(double *A, const unsigned int NUMELEM, double *globalSum);
__global__ void computeSumDiffElemWithOneElemPerThreadWithSM(double *A, const unsigned int NUMELEM,  double *globalSum, double *globalSumDiffElemTotalSumSquared);


//Modes 2, 3, 4
__global__ void computeStdDevWithGlobalSumDiffElem(const unsigned int NUMELEM, double *globalSumDiffElemTotalSumSquared,  double *stddev);

int main(int argc, char *argv[])
{
	
	//change OpenMP settings
	//disregard --- only for parallel CPU version if applicable
	omp_set_num_threads(1);

  	//seed random number generator with constant seed
  	//Do not change this or you will be unable to validate your program output
  	srand(123);  

  	//input arrays
  	double * A;
 
  	A=(double *)malloc(sizeof(double)*N);
 
  	//init input array of FP64 with random numbers between 0 and 1
  	for (unsigned int i=0; i<N; i++){
  		A[i]=(double)rand()/(double)RAND_MAX;
  	}

	printf("\nMemory requested for 1 array of length N (GiB) %f", (N*sizeof(double)/(1024.0*1024.0*1024.0)));



	///////////////////////////
	//CPU version:
	///////////////////////////

	//Compute the standard deviation multiplication on the CPU
	double stddevCPU = computeStdDevCPU(A, N);

	printf("\nStd. Dev. CPU: %f", stddevCPU);

	/////////////////////////////
	//GPU
	////////////////////////////	

	
	double tstart=omp_get_wtime();

	double * dev_A;
	double * dev_mean;
	double * dev_stddev;

	double meanGPU = 0;

	//the stddev computed on the GPU that will be copied back to the host
	double stddevGPU; 
	
	//allocate and copy to the device: A, mean
	//write code here
	



	//allocate stddev on device (this is the output)
	gpuErrchk(cudaMalloc((double**)&dev_stddev, sizeof(double)));	


	//allocate the global sum and difference between each element and the mean 
	//for Modes 2-3 (and 4 if applicable):
	double * dev_globalSum;
	double * dev_globalSumDiffElemTotalSumSquared;
	double globalSum=0;
	double * globalSumDiffElemTotalSumSquared=0;
	//Modes 2-4
	if(MODE!=1)
	{
		//write code here

		//allocate and copy to GPU
	}

	

	//execute kernels based on MODE
	//MODE==1
	//One thread on the GPU does all work
	if (MODE==1){
		printf("\nMODE==1");
		//set number of blocks
		dim3 dimGrid(1, 1, 1); 
		dim3 dimBlock(1, 1, 1);
		
		computeMeanOneThread<<<dimGrid, dimBlock>>>(dev_A, dev_mean, N);
		computeStdDevOneThread<<<dimGrid, dimBlock>>>(dev_A, dev_mean, N, dev_stddev);

	}
	//MODE==2
	//One element is computed per thread
	//atomic updates to global memory
	else if (MODE==2){
		printf("\nMODE==2");
		
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 
		
		//Write code here
		//Uncomment and update code below
		//unsigned int numBlocks = 0;

		// dim3 dimGrid(numBlocks, 1, 1);
		// dim3 dimBlock(BLOCKDIM, 1, 1);

		//Compute Step 1 -- sum of all elements
		// computeGlobalSumWithOneElemPerThread<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum);
		//Compute Step 2 -- compute mean, and then the sum of differences between each element and the mean
		// computeSumDiffElemWithOneElemPerThread<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum, dev_globalSumDiffElemTotalSumSquared);
		//Compute Step 3 -- standard deviation with one thread
		// computeStdDevWithGlobalSumDiffElem<<<1, 1>>>(N, dev_globalSumDiffElemTotalSumSquared,  dev_stddev);
	}
	//MODE==3
	//Multiple elements are computed per thread
	//atomic updates to global memory 
	else if (MODE==3){
		printf("\nMODE==3");
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 
		
		//Write code here
		//Uncomment and update code below
		//unsigned int numBlocks = 0;

		// dim3 dimGrid(numBlocks, 1, 1);
		// dim3 dimBlock(BLOCKDIM, 1, 1);

		//Compute Step 1 -- sum of all elements
		// computeGlobalSumWithMultipleElemsPerThread<<<dimGrid, dimBlock>>>(dev_A, N, NUMELEMPERTHREAD, dev_globalSum);
		//Compute Step 2 -- compute mean, and then the sum of differences between each element and the mean
		// computeSumDiffElemWithMultipleElemsPerThread<<<dimGrid, dimBlock>>>(dev_A, N, NUMELEMPERTHREAD, dev_globalSum, dev_globalSumDiffElemTotalSumSquared);
		//Compute Step 3 -- standard deviation with one thread
		// computeStdDevWithGlobalSumDiffElem<<<1, 1>>>(N, dev_globalSumDiffElemTotalSumSquared,  dev_stddev);	

	}
	//MODE==4
	//Uses shared memory to compute the standard deviation
	//The total number of threads is N
	//Each thread in a block writes its values to shared memory
	//And one thread in the block writes the block values to global memory
	else if (MODE==4){
		printf("\nMODE==4");
			
		//set number of blocks
		unsigned int BLOCKDIM = BLOCKSIZE; 

		//Write code here
		//Uncomment and update code below
		//unsigned int numBlocks = 0;

		// dim3 dimGrid(numBlocks, 1, 1);
		// dim3 dimBlock(BLOCKDIM, 1, 1);

		//Compute Step 1 -- sum of all elements
		// computeGlobalSumWithOneElemPerThreadWithSM<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum);
		//Compute Step 2 -- compute mean, and then the sum of differences between each element and the mean
		// computeSumDiffElemWithOneElemPerThreadWithSM<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum, dev_globalSumDiffElemTotalSumSquared);
		//Compute Step 3 -- standard deviation with one thread
		// computeStdDevWithGlobalSumDiffElem<<<1, 1>>>(N, dev_globalSumDiffElemTotalSumSquared,  dev_stddev);

	}
	else
	{
		printf("\nInvalid Mode! Mode: %d", MODE);
		return 0;
	}
	

	//check kernel errors
	gpuErrchk(cudaGetLastError());

	//end execute kernel

	//Copy stddev from the GPU to the host
	
	//write code here
	
	double tend=omp_get_wtime();
	
	printf("\nStd. Dev. GPU: %f", stddevGPU);
	printf("\nTime to compute Std. Dev. on the GPU: %f", tend - tstart);

	printf("\n[MODE: %d, N: %d] Difference between GPU and CPU Std. Dev: %f", MODE, N, abs(stddevCPU-stddevGPU));
	


	//free memory

	//Uncomment and update as needed
	// free(A);
  	
  	// cudaFree(dev_A);
  	// cudaFree(dev_mean);
  	// cudaFree(dev_stddev);

  	// if(MODE!=1)
  	// {
  	// 	cudaFree(dev_globalSum);
	// 	cudaFree(dev_globalSumDiffElemTotalSumSquared);
  	// }

	printf("\n\n");

	return 0;
}







///////////////////////////////////
//Sequential CPU Code
///////////////////////////////////


double computeStdDevCPU(double * A, const unsigned int NUMELEM)
{

	//write code here (below)

	
	double tstart=omp_get_wtime();

	double stddev=0;
	//Step 1: Compute the mean


	//Step 2: Compute Sum of difference between each element and the mean
	

	//Step 3: Compute std dev
	

	double tend=omp_get_wtime();
	printf("\nTime to compute Std. Dev. on the CPU: %f", tend - tstart);

	return stddev;
}

///////////////////////////////////
//End Sequential CPU Code
///////////////////////////////////

///////////////////////////////////
//GPU Mode 1
///////////////////////////////////

__global__ void computeMeanOneThread(double *A, double *mean, const unsigned int NUMELEM) {

//Step 1: Compute the mean

//write code here



return;
}

__global__ void computeStdDevOneThread(double *A, double *mean, const unsigned int NUMELEM, double *stddev) {

//Step 2: Compute Sum of difference between each element and the mean

//write code here

//Step 3: Compute std dev

//write code here

return;
}


///////////////////////////////////
//End GPU Mode 1
///////////////////////////////////


///////////////////////////////////
//GPU Mode 2
///////////////////////////////////

__global__ void computeGlobalSumWithOneElemPerThread(double *A, const unsigned int NUMELEM, double *globalSum) {

//Step 1: Compute the sum of all elements -- assign multiple elements per thread

//write code here

return;
}

__global__ void computeSumDiffElemWithOneElemPerThread(double *A, const unsigned int NUMELEM,  double *globalSum, double *globalSumDiffElemTotalSumSquared)
{

//write code here

return;
}

__global__ void computeStdDevWithGlobalSumDiffElem(const unsigned int NUMELEM, double *globalSumDiffElemTotalSumSquared,  double *stddev)
{

//write code here

return;
}


///////////////////////////////////
//End GPU Mode 2
///////////////////////////////////


///////////////////////////////////
//GPU Mode 3
///////////////////////////////////

__global__ void computeGlobalSumWithMultipleElemsPerThread(double *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD, double *globalSum) {

//Step 1: Compute the sum of all elements -- assign multiple elements per thread

//write code here

return;
}





__global__ void computeSumDiffElemWithMultipleElemsPerThread(double *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD,  double *globalSum, double *globalSumDiffElemTotalSumSquared)
{

//write code here

return;
}




///////////////////////////////////
//End GPU Mode 3
///////////////////////////////////

///////////////////////////////////
//GPU Mode 4
///////////////////////////////////




__global__ void computeGlobalSumWithOneElemPerThreadWithSM(double *A, const unsigned int NUMELEM, double *globalSum) {

	//Step 1: Compute the sum of all elements
	
	//write code here

	return;
}

__global__ void computeSumDiffElemWithOneElemPerThreadWithSM(double *A, const unsigned int NUMELEM,  double *globalSum, double *globalSumDiffElemTotalSumSquared)
{

	//write code here

	return;
}

///////////////////////////////////
//End GPU Mode 4
///////////////////////////////////















