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
#define N 1000000  //Elements in N
#define MODE 3 //GPU Mode
#define NUMELEMPERTHREAD 8 //This is r in the assignment instructions for MODE 3
#define BLOCKSIZE 1024 //GPU CUDA block size for the first two kernels for MODES 2-4.


using namespace std;


//function prototypes

//CPU function that computes the standard deviation
float computeStdDevCPU(float * A, const unsigned int NUMELEM);

//GPU kernels:
//Mode 1:
__global__ void computeMeanOneThread(float *A, float *mean, const unsigned int NUMELEM);
__global__ void computeStdDevOneThread(float *A, float *mean, const unsigned int NUMELEM, float *stddev);


//Mode 2:
__global__ void computeGlobalSumWithOneElemPerThread(float *A, const unsigned int NUMELEM, float *globalSum);
__global__ void computeSumDiffElemWithOneElemPerThread(float *A, const unsigned int NUMELEM, float *globalSum, float *globalSumDiffElemTotalSumSquared);


//Mode 3:
__global__ void computeGlobalSumWithMultipleElemsPerThread(float *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD, float *globalSum);
__global__ void computeSumDiffElemWithMultipleElemsPerThread(float *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD,  float *globalSum, float *globalSumDiffElemTotalSumSquared);


//Mode 4:
__global__ void computeGlobalSumWithOneElemPerThreadWithSM(float *A, const unsigned int NUMELEM, float *globalSum);
__global__ void computeSumDiffElemWithOneElemPerThreadWithSM(float *A, const unsigned int NUMELEM,  float *globalSum, float *globalSumDiffElemTotalSumSquared);


//Modes 2, 3, 4
__global__ void computeStdDevWithGlobalSumDiffElem(const unsigned int NUMELEM, float *globalSumDiffElemTotalSumSquared,  float *stddev);

int main(int argc, char *argv[])
{
	
	//change OpenMP settings
	//disregard --- only for parallel CPU version if applicable
	omp_set_num_threads(1);

  	//seed random number generator with constant seed
  	//Do not change this or you will be unable to validate your program output
  	srand(123);  

  	//input arrays
  	float * A;
 
  	A=(float *)malloc(sizeof(float)*N);
 
  	//init input array of FP64 with random numbers between 0 and 1
  	for (unsigned int i=0; i<N; i++){
  		A[i]=(float)rand()/(float)RAND_MAX;
  	}
	printf("\nMemory requested for 1 array of length N (GiB) %f", (N*sizeof(float)/(1024.0*1024.0*1024.0)));



	///////////////////////////
	//CPU version:
	///////////////////////////

	//Compute the standard deviation multiplication on the CPU
	float stddevCPU = computeStdDevCPU(A, N);

	printf("\nStd. Dev. CPU: %f", stddevCPU);

	/////////////////////////////
	//GPU
	////////////////////////////	

	
	float tstart=omp_get_wtime();

	float * dev_A;
	float * dev_mean;
	float * dev_stddev;

	float meanGPU = 0;

	//the stddev computed on the GPU that will be copied back to the host
	float stddevGPU ; 
	
	//allocate and copy to the device: A, mean
	//write code here
	cudaError_t errCode = cudaSuccess;
	errCode =cudaMalloc((float**)&dev_A,N*sizeof(float));
	if(errCode != cudaSuccess )
	{
		cout << "\nError with Array A" << errCode << "\n";
	}
	errCode = cudaMemcpy(dev_A,A,sizeof(float) * N, cudaMemcpyHostToDevice);

	if(errCode != cudaSuccess ) cout << "\nError wth hst" << errCode << "\n";
	gpuErrchk(cudaMalloc((float**)&dev_mean,sizeof(float)));	
	errCode = cudaMemcpy(dev_mean,&meanGPU,sizeof(float),cudaMemcpyHostToDevice);
	gpuErrchk(cudaMalloc((float**)&dev_stddev, sizeof(float)));	


	//allocate the global sum and difference between ealement and the mean 
	//for Modes 2-3 (and 4 if applicable):
	float * dev_globalSum;
	float * dev_globalSumDiffElemTotalSumSquared;
	float globalSum=0;
	float globalSumDiffElemTotalSumSquared = 0;
	//Modes 2-4
	if(MODE!=1)
	{
		//write code here
		gpuErrchk(cudaMalloc((float **)&dev_globalSum,sizeof(float))); 
		gpuErrchk(cudaMalloc((float **)&dev_globalSumDiffElemTotalSumSquared,sizeof(float))); 
		//allocate and copy to GPU
		gpuErrchk(cudaMemcpy(dev_globalSum,&globalSum,sizeof(float),cudaMemcpyHostToDevice));  	
		gpuErrchk(cudaMemcpy(dev_globalSumDiffElemTotalSumSquared,&globalSumDiffElemTotalSumSquared,sizeof(float),cudaMemcpyHostToDevice));	
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
		unsigned const int numBlocks = ceil(N *1.0 / BLOCKSIZE) ;
		printf("\n %d blocks",numBlocks);
		 dim3 dimGrid(numBlocks, 1, 1);
		 dim3 dimBlock(BLOCKDIM, 1, 1);

		//Compute Step 1 -- sum of all elements
		 computeGlobalSumWithOneElemPerThread<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum);
	cudaDeviceSynchronize();	
	//Compute Step 2 -- compute mean, and then the sum of differences between each element and the mean
		 computeSumDiffElemWithOneElemPerThread<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum, dev_globalSumDiffElemTotalSumSquared);
	cudaDeviceSynchronize();	
	//Compute Step 3 -- standard deviation with one thread
		 computeStdDevWithGlobalSumDiffElem<<<1, 1>>>(N, dev_globalSumDiffElemTotalSumSquared,  dev_stddev);
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
		unsigned int numBlocks = ceil(N* 1.0/NUMELEMPERTHREAD/(BLOCKSIZE)); 
		printf("\n %d num of blocks ",numBlocks); 
		 dim3 dimGrid(numBlocks, 1, 1);
		 dim3 dimBlock(BLOCKDIM, 1, 1);

		//Compute Step 1 -- sum of all elements
		 computeGlobalSumWithMultipleElemsPerThread<<<dimGrid, dimBlock>>>(dev_A, N, NUMELEMPERTHREAD, dev_globalSum);
		//Compute Step 2 -- compute mean, and then the sum of differences between each element and the mean
		 computeSumDiffElemWithMultipleElemsPerThread<<<dimGrid, dimBlock>>>(dev_A, N, NUMELEMPERTHREAD, dev_globalSum, dev_globalSumDiffElemTotalSumSquared);
		//Compute Step 3 -- standard deviation with one thread
		 computeStdDevWithGlobalSumDiffElem<<<1, 1>>>(N, dev_globalSumDiffElemTotalSumSquared,  dev_stddev);	

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
		unsigned const int numBlocks =ceil(N * 1.0/BLOCKSIZE);

		 dim3 dimGrid(numBlocks, 1, 1);
		 dim3 dimBlock(BLOCKDIM, 1, 1);

		//Compute Step 1 -- sum of all elements
		 computeGlobalSumWithOneElemPerThreadWithSM<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum);
		//Compute Step 2 -- compute mean, and then the sum of differences between each element and the mean
		 computeSumDiffElemWithOneElemPerThreadWithSM<<<dimGrid, dimBlock>>>(dev_A, N, dev_globalSum, dev_globalSumDiffElemTotalSumSquared);
		//Compute Step 3 -- standard deviation with one thread
		 computeStdDevWithGlobalSumDiffElem<<<1, 1>>>(N, dev_globalSumDiffElemTotalSumSquared,  dev_stddev);

	}
	else
	{
		printf("\nInvalid Mode! Mode: %d", MODE);
		return 0;
	}
	
	cudaDeviceSynchronize(); 
	//check kernel errors
	gpuErrchk(cudaGetLastError());

	//end execute kernel

	//Copy stddev from the GPU to the host
	//write code here
	errCode = cudaMemcpy(&stddevGPU,dev_stddev,sizeof(float),cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess ) cout << "gpu not get stddev: " << errCode << "\n";	
	float tend=omp_get_wtime();
	
	printf("\nStd. Dev. GPU: %f", stddevGPU);
	printf("\nTime to compute Std. Dev. on the GPU: %f", tend - tstart);

	printf("\n[MODE: %d, N: %d] Difference between GPU and CPU Std. Dev: %f", MODE, N, abs(stddevCPU-stddevGPU));
	


	//free memory

	//Uncomment and update as needed
	 free(A);
  	
  	 cudaFree(dev_A);
  	 cudaFree(dev_mean);
  	 cudaFree(dev_stddev);

  	 if(MODE!=1)
  	 {
  	 	cudaFree(dev_globalSum);
	 	cudaFree(dev_globalSumDiffElemTotalSumSquared);
  	 }

	printf("\n\n");

	return 0;
}







///////////////////////////////////
//Sequential CPU Code
///////////////////////////////////


float computeStdDevCPU(float * A, const unsigned int NUMELEM)
{

	//write code here (below)

	
	float tstart=omp_get_wtime();

	float stddev=0;
	float sum = 0;
	//Step 1: Compute the mean
	for(int i = 0 ; i < NUMELEM ; i++ )
	{
		sum+=A[i]; 
	}
	sum = sum/NUMELEM;
	float diff = 0 ;
	//Step 2: Compute Sum of difference between each element and the mean
	for(int i = 0 ; i < NUMELEM ; i++ )
	{
		diff += pow(A[i] - sum , 2 );

	}	
	diff = diff / NUMELEM ;
	//Step 3: Compute std dev
	stddev = sqrt(diff);	

	float tend=omp_get_wtime();
	printf("\nTime to compute Std. Dev. on the CPU: %f", tend - tstart);

	return stddev;
}

///////////////////////////////////
//End Sequential CPU Code
///////////////////////////////////

///////////////////////////////////
//GPU Mode 1
///////////////////////////////////

__global__ void computeMeanOneThread(float *A, float *mean, const unsigned int NUMELEM) {

//Step 1: Compute the mean
for(int i = 0 ; i < NUMELEM ; i ++ ) {
	atomicAdd(mean,A[i]);
}
	 *mean /= NUMELEM;
return;
}

__global__ void computeStdDevOneThread(float *A, float *mean, const unsigned int NUMELEM, float *stddev) {

//Step 2: Compute Sum of difference between each element and the mean
float sqr = 0 ;

//write code here
for(int i = 0 ; i < NUMELEM ; i++) {

	sqr=pow(A[i] - *mean,2) ;
	atomicAdd(stddev,sqr);
}
//Step 3: Compute std dev
//write code here



	*stddev /= NUMELEM;
	*stddev  = sqrt(*stddev);

return;
}


///////////////////////////////////
//End GPU Mode 1
///////////////////////////////////


///////////////////////////////////
//GPU Mode 2
///////////////////////////////////

__global__ void computeGlobalSumWithOneElemPerThread(float *A, const unsigned int NUMELEM, float *globalSum) {

//Step 1: Compute the sum of all elements -- assign multiple elements per thread

//write code here
unsigned const int tid = threadIdx.x + (blockIdx.x * blockDim.x);
if(tid < NUMELEM ) {
	atomicAdd(globalSum,A[tid]);
}
return;
}

__global__ void computeSumDiffElemWithOneElemPerThread(float *A, const unsigned int NUMELEM,  float *globalSum, float *globalSumDiffElemTotalSumSquared)
{

//write code here
unsigned const int tid = threadIdx.x + (blockIdx.x * blockDim.x);
if(tid < NUMELEM ) {
	float mean = *globalSum/ NUMELEM ;
	float sqr = pow(A[tid] - mean,2);
	atomicAdd(globalSumDiffElemTotalSumSquared,sqr);
}
return;
}

__global__ void computeStdDevWithGlobalSumDiffElem(const unsigned int NUMELEM, float *globalSumDiffElemTotalSumSquared,  float *stddev)
{
//write code here
	float diff = *globalSumDiffElemTotalSumSquared / NUMELEM ;
	*stddev = sqrt(diff);
return;
}


///////////////////////////////////
//End GPU Mode 2
///////////////////////////////////


///////////////////////////////////
//GPU Mode 3
///////////////////////////////////

__global__ void computeGlobalSumWithMultipleElemsPerThread(float *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD, float *globalSum) {

//Step 1: Compute the sum of all elements -- assign multiple elements per thread
float sum = 0;
//write code here
unsigned const int tid = threadIdx.x + (blockIdx.x * blockDim.x );
for(int i = 0 ; i< ELEMPERTHREAD ; i++ )
{
	int id = tid +  i * blockDim.x * gridDim.x;
	if(id < NUMELEM ){
		sum+=A[id];	
	}
}
atomicAdd(globalSum,sum);
return;
}





__global__ void computeSumDiffElemWithMultipleElemsPerThread(float *A, const unsigned int NUMELEM, const unsigned int ELEMPERTHREAD,  float *globalSum, float *globalSumDiffElemTotalSumSquared)
{

//write code here
unsigned const tid = threadIdx.x + ( blockIdx.x * blockDim.x);
float diff = 0 ;
float mean = *globalSum / NUMELEM ;
for( int i = 0 ; i < ELEMPERTHREAD  ; i++ ) {
	int id = tid + i * blockDim.x * gridDim.x ;
	if(id < NUMELEM )  {
		diff+= pow(A[id] - mean,2);
	}
}
	atomicAdd(globalSumDiffElemTotalSumSquared,diff);


return;
}




///////////////////////////////////
//End GPU Mode 3
///////////////////////////////////

///////////////////////////////////
//GPU Mode 4
///////////////////////////////////




__global__ void computeGlobalSumWithOneElemPerThreadWithSM(float *A, const unsigned int NUMELEM, float *globalSum) {

	//Step 1: Compute the sum of all elements
	
	//write code here
	unsigned const int tid = threadIdx.x + ( blockIdx.x * blockDim.x );

	__shared__ float sum ;
	if(threadIdx.x ==  0 ) {
		sum = 0;
	}
	__syncthreads();

	if(tid < NUMELEM ) { 
		atomicAdd(&sum , A[tid] );
	}
	
	__syncthreads();

	if(threadIdx.x == 0 ) {
		atomicAdd(globalSum,sum);
	}

	return;
}

__global__ void computeSumDiffElemWithOneElemPerThreadWithSM(float *A, const unsigned int NUMELEM,  float *globalSum, float *globalSumDiffElemTotalSumSquared)
{

	//write code here
	unsigned const int tid = threadIdx.x + ( blockIdx.x * blockDim.x ) ;
	
	__shared__  float mean ;

	__shared__ float diff ;

	if(threadIdx.x == 0 ) {
		diff = 0 ;
		mean = *globalSum / NUMELEM ;
	}

	__syncthreads();


	if(tid < NUMELEM ) {
		atomicAdd(&diff,pow(A[tid]-mean,2));
	}

	__syncthreads();

	if(threadIdx.x == 0 ) { 
		atomicAdd(globalSumDiffElemTotalSumSquared,diff);
	}
	return;
}

///////////////////////////////////
//End GPU Mode 4
///////////////////////////////////















