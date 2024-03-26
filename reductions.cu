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

//Modes -1, -2, -3 are the single block kernels from the textbook

//Modes 1, 2, 3 are multiblock kernels

#define MODE 3
#define BLOCKSIZE 512

//Only for mode 3
#define COARSEFACTOR 16 

using namespace std;


//function prototypes
void warmUpGPU();
void checkParams(const unsigned int N);
void generateDataset(int * A, const unsigned int N);
void computeReductionCPU(int * A, const unsigned int N, int * globalSum);



//Modes -1, -2, -3: these are single block kernels in the textbook
__global__ void sumReductionGlobalMemoryOneBlockKernel(int * A, const unsigned int N, int * outputSum);
__global__ void sumReductionGlobalMemoryOneBlockLessDivergenceKernel(int * A, const unsigned int N, int * outputSum);
__global__ void sumReductionSharedMemoryOneBlockKernel(int * A, const unsigned int N, int * outputSum);

//Mode 1
__global__ void inefficientReductionKernel(int * A, const unsigned int N, int * outputSum);

//Mode 2
__global__ void sumReductionSharedMemoryMultipleBlockKernel(int * A, const unsigned int N, int * outputSum);

//Mode 3
__global__ void sumReductionSharedMemoryMultipleBlockThreadCoarseningKernel(int * A, const unsigned int N, int * outputSum);

//Modes 4 and 5 are for the assignment



int main(int argc, char *argv[])
{
  warmUpGPU(); 
  
  unsigned int N=0;

  if (argc != 2) {
    fprintf(stderr,"Please provide the following on the command line: N.\n");
    exit(0);
  }

  sscanf(argv[1],"%d",&N);
 
  checkParams(N);

  printf("\nAllocating the following amount of memory for the dataset: %f GiB", (sizeof(int)*N)/(1024*1024*1024.0));
  int * A=(int *)malloc(sizeof(int)*N);
  generateDataset(A, N);
  
  //output of the reduction on the GPU
  int globalSum=0;
  
  //output of the reduction on the CPU (for validation purposes)
  int globalSumCPU=0;
  
  //Compute on the CPU:
  computeReductionCPU(A, N, &globalSumCPU);
  printf("\nCPU global sum (sequential CPU algorithm): %d", globalSumCPU);

  
  //CUDA event timers (outputs time in ms not s)
  float totalKernelTime;
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);


  double tstart=omp_get_wtime();

  int * dev_A;
  //allocate on the device: A
  gpuErrchk(cudaMalloc((int**)&dev_A, sizeof(int)*N));  
  
  //copy A to device
  gpuErrchk(cudaMemcpy(dev_A, A, sizeof(int)*N, cudaMemcpyHostToDevice));
  
  
  int * dev_globalSum;
  //allocate on the device: the result
  gpuErrchk(cudaMalloc((int**)&dev_globalSum, sizeof(int)));  

  //copy initialized reduction value to device
  gpuErrchk(cudaMemcpy(dev_globalSum, &globalSum, sizeof(int), cudaMemcpyHostToDevice));


  cudaEventRecord(begin,0);

  //Textbook modes with one block
  //These will not be used in the assignment
  if(MODE==-1 || MODE==-2 || MODE==-3)
  {
    if(N!=2*BLOCKSIZE)
    {
      printf("\nError: N should be constrained to 2*BLOCKSIZE\n");
      return 0;
    }  

    if(MODE==-1)
    {
      sumReductionGlobalMemoryOneBlockKernel<<<1, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
    }
    else if(MODE==-2)
    {
      sumReductionGlobalMemoryOneBlockLessDivergenceKernel<<<1, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
    }
    else if(MODE==-3)
    {
      sumReductionSharedMemoryOneBlockKernel<<<1, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
    }

  }
  


  //Textbook modes: multi-block
  //Multi-block -- atomic updates only
  //Inefficient reduction
  if(MODE==1)  {
  const unsigned int NBLOCKS=ceil(N*1.0/BLOCKSIZE*1.0);
  printf("\nNum blocks: %u", NBLOCKS);
  inefficientReductionKernel<<<NBLOCKS,BLOCKSIZE>>>(dev_A, N, dev_globalSum);
  }
  //Multi-block -- no thread coarsening
  else if(MODE==2){
  unsigned int NBLOCKS = ceil(N*1.0/(BLOCKSIZE*2.0));
  printf("\nNum blocks: %u", NBLOCKS);
  sumReductionSharedMemoryMultipleBlockKernel<<<NBLOCKS, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
  }
  //Multi-block -- with thread coarsening
  else if(MODE==3){
  unsigned int NBLOCKS = ceil(N*1.0/(BLOCKSIZE*2.0*COARSEFACTOR));
  printf("\nNum blocks: %u", NBLOCKS);
  sumReductionSharedMemoryMultipleBlockThreadCoarseningKernel<<<NBLOCKS, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
  }



  
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&totalKernelTime,begin,end);
  
  printf("\n[N: %u, MODE: %d, BLOCKSIZE: %d, COARSEFACTOR: %d] Kernel-only time to perform GPU reduction (ms): %f", N, MODE, BLOCKSIZE, COARSEFACTOR, totalKernelTime);
  
  gpuErrchk(cudaMemcpy(&globalSum, dev_globalSum, sizeof(int), cudaMemcpyDeviceToHost));

  double tend=omp_get_wtime();
  printf("\nTotal time to perform GPU reduction (s): %f", tend - tstart);


  printf("\nGPU global sum: %d", globalSum);


  free(A);
  printf("\n\n");

  return 0;
}




void warmUpGPU()
{
  cudaDeviceSynchronize();
}


//generates data in the range [-4, 9]
void generateDataset(int * A, const unsigned int N)
{
  //seed random number generator
  srand(493);  
  const int minValue = -5;
  const int maxValue = 10;
  const unsigned int range = abs(minValue)+abs(maxValue);
  for (unsigned int i=0; i<N; i++){
    A[i]=minValue+((float)rand()/(float)RAND_MAX)*range;
  }
}


void checkParams(const unsigned int N)
{
  if(N%BLOCKSIZE!=0){
    printf("\nError: N does not evenly divide the block size: N: %u, Block Size: %d\n\n", N, BLOCKSIZE);
    exit(0);
  }

  if(N>536870912){
    printf("\nError: N>536870912. A significantly larger value of N will risk overflowing the global sum.\n\n");
    exit(0);
  }
}


void computeReductionCPU(int * A, const unsigned int N, int * globalSum)
{
    double tstart=omp_get_wtime();
    
      for(unsigned int i=0; i<N; i++){
          *globalSum+=A[i];
      }
  
    double tend=omp_get_wtime();
    
    printf("\nTime to compute the reduction on the CPU (sequential): %f", tend - tstart);
}




//Mode -1 
//Uses a single CUDA block
//Sum reduction kernel in Figure 10.6 of the textbook
__global__ void sumReductionGlobalMemoryOneBlockKernel(int * A, const unsigned int N, int * outputSum) {

  unsigned int tid = threadIdx.x*2;

  for (unsigned int stride=1; stride<=blockDim.x; stride*=2)
  {
    if(threadIdx.x%stride == 0){
    A[tid] += A[tid+stride];
    }

    //syncthreads because we will be reading after writing global memory
    __syncthreads();
  }

  if(threadIdx.x==0){

    *outputSum = A[0];
  }
 
  return;
}

//Mode -2
//Uses a single CUDA block
//Sum reduction kernel in Figure 10.9 of the textbook
__global__ void sumReductionGlobalMemoryOneBlockLessDivergenceKernel(int * A, const unsigned int N, int * outputSum) {

  unsigned int tid = threadIdx.x;

  //start with maximum stride (elements to add are blockDim.x elements away at first iteration)
  //next iteration is a half stride away and so on
  for (unsigned int stride = blockDim.x; stride>=1; stride/=2)
  {
    if(threadIdx.x<stride){
    A[tid] += A[tid+stride];
    }

    //syncthreads because we will be reading after writing global memory
    __syncthreads();
  }

  if(threadIdx.x==0){

    *outputSum = A[0];
  }

  return;
}


//Mode -3
//Uses a single CUDA block
//Sum reduction kernel in Figure 10.11 of the textbook
__global__ void sumReductionSharedMemoryOneBlockKernel(int * A, const unsigned int N, int * outputSum) {

  __shared__ int sharedA[BLOCKSIZE];
  
  unsigned int tid = threadIdx.x;

  //First iteration is read from global memory and written to shared memory
  sharedA[tid]=A[tid]+A[tid+BLOCKSIZE];

  

  //Start shared memory stride
  //start with maximum stride (elements to add are blockDim.x elements away at first iteration)
  //next iteration is a half stride away and so on
  for (unsigned int stride = blockDim.x/2; stride>=1; stride/=2)
  {

    //syncthreads because we will be reading after writing global memory
    __syncthreads();

    if(threadIdx.x<stride){
    sharedA[tid] += sharedA[tid+stride];
    }

  
  }

  if(threadIdx.x==0){

    *outputSum = sharedA[0];
  }
  
  return;
}

//Mode 1
//"inefficient reduction" 
//N threads and only atomic updates to the global sum
__global__ void inefficientReductionKernel(int * A, const unsigned int N, int * globalSum) {

  int tid = threadIdx.x + (blockIdx.x*blockDim.x);

  if(tid<N){
      atomicAdd(globalSum, A[tid]);
  }

  return;
}



//Mode 2
//Sum reduction kernel in Figure 10.13 of the textbook
//Allows for multiple blocks (instead of one block)
//Uses N/2 threads
__global__ void sumReductionSharedMemoryMultipleBlockKernel(int * A, const unsigned int N, int * outputSum) {

  __shared__ int sharedA[BLOCKSIZE];
  
  //starting position in A based on block ID
  //called "segment" in the textbook
  unsigned int offsetIdx = 2*blockDim.x*blockIdx.x;

  unsigned int offsetIdxThread = offsetIdx + threadIdx.x;

  //First iteration is read from global memory and written to shared memory
  sharedA[threadIdx.x]=A[offsetIdxThread]+A[offsetIdxThread+BLOCKSIZE];

  //Start shared memory stride
  //start with maximum stride (elements to add are blockDim.x elements away at first iteration)
  //next iteration is a half stride away and so on
  for (unsigned int stride = blockDim.x/2; stride>=1; stride/=2)
  {

    //syncthreads because we will be reading after writing shared memory
    __syncthreads();

    if(threadIdx.x<stride){
    sharedA[threadIdx.x] += sharedA[threadIdx.x+stride];
    }

  
  }

  if(threadIdx.x==0){
    atomicAdd(outputSum, sharedA[0]);
  }

  return;
}


//Mode 3
//Sum reduction kernel in Figure 10.15 of the textbook
//Allows for multiple blocks (instead of one block)
//Uses fewer than N/2 threads
__global__ void sumReductionSharedMemoryMultipleBlockThreadCoarseningKernel(int * A, const unsigned int N, int * outputSum) {

__shared__ int sharedA[BLOCKSIZE];
  
  //starting position in A based on block ID
  //called "segment" in the textbook
  unsigned int offsetIdx = 2*COARSEFACTOR*blockDim.x*blockIdx.x;

  unsigned int offsetIdxThread = offsetIdx + threadIdx.x;

  int localSum = A[offsetIdxThread];

  for(unsigned int i=1; i<COARSEFACTOR*2; i++){
  localSum += A[offsetIdxThread + i*blockDim.x];
  }


  sharedA[threadIdx.x] = localSum;  

  //Start shared memory stride
  //start with maximum stride (elements to add are blockDim.x elements away at first iteration)
  //next iteration is a half stride away and so on
  for (unsigned int stride = blockDim.x/2; stride>=1; stride/=2)
  {

    //syncthreads because we will be reading after writing shared memory
    __syncthreads();

    if(threadIdx.x<stride){
    sharedA[threadIdx.x] += sharedA[threadIdx.x+stride];
    }

  }

  if(threadIdx.x==0){
    atomicAdd(outputSum, sharedA[0]);
  }

  return;
}

