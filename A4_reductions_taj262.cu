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


//#define MODE 3
//#define BLOCKSIZE 512

//Only for mode 3,4,5
//#define COARSEFACTOR 16 

using namespace std;


//function prototypes
void warmUpGPU();
void checkParams(const unsigned int N);
void generateDataset(int * A, const unsigned int N);
void computeReductionCPU(int * A, const unsigned int N, int * globalSum);


//Mode 1
__global__ void inefficientReductionKernel(int * A, const unsigned int N, int * outputSum);

//Mode 2
__global__ void sumReductionSharedMemoryMultipleBlockKernel(int * A, const unsigned int N, int * outputSum);

//Mode 3
__global__ void sumReductionSharedMemoryMultipleBlockThreadCoarseningKernel(int * A, const unsigned int N, int * outputSum);

//Use atomic updates to shared memory:
//Mode 4
__global__ void extendModeThreeWithAtomicUpdatesSharedMemoryOneElems(int * A, const unsigned int N, int * outputSum);

//Mode 5
__global__ void extendModeThreeWithAtomicUpdatesSharedMemoryNumWarpsElems(int * A, const unsigned int N, int * outputSum);



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
  float totalKernelTime,totalTransferTime;
  cudaEvent_t begin, end;
  cudaEvent_t dev_begin,dev_end;
  cudaEventCreate(&dev_begin);
  cudaEventCreate(&dev_end);
  cudaEventCreate(&begin);
  cudaEventCreate(&end);


  double tstart=omp_get_wtime();

  int * dev_A;
  //allocate on the device: A
  gpuErrchk(cudaMalloc((int**)&dev_A, sizeof(int)*N));  
  
  //write code
  cudaEventRecord(dev_begin,0);
  //copy A to device
  gpuErrchk(cudaMemcpy(dev_A, A, sizeof(int)*N, cudaMemcpyHostToDevice));
  //write code
   cudaEventRecord(dev_end,0);
  cudaEventSynchronize(dev_end);
  cudaEventElapsedTime(&totalTransferTime,dev_begin,dev_end);
  
  printf("\nTotal Transfer Time to Dev A (ms): %f",totalTransferTime);
  int * dev_globalSum;
  //allocate on the device: the result
  gpuErrchk(cudaMalloc((int**)&dev_globalSum, sizeof(int)));  

  //copy initialized reduction value to device
  gpuErrchk(cudaMemcpy(dev_globalSum, &globalSum, sizeof(int), cudaMemcpyHostToDevice));


  cudaEventRecord(begin,0);

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
  //Multi-block -- with thread coarsening -- with shared-memory atomic updates   
  //One element in shared memory per block
  else if(MODE==4){
  unsigned int NBLOCKS = ceil(N*1.0/(BLOCKSIZE*2.0*COARSEFACTOR));
  printf("\nNum blocks: %u", NBLOCKS);
  extendModeThreeWithAtomicUpdatesSharedMemoryOneElems<<<NBLOCKS, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
  }
  //Multi-block -- with thread coarsening -- with shared-memory atomic updates
  //One element in shared memory per warp per block
  else if(MODE==5){
  unsigned int NBLOCKS = ceil(N*1.0/(BLOCKSIZE*2.0*COARSEFACTOR));
  printf("\nNum blocks: %u", NBLOCKS);
  extendModeThreeWithAtomicUpdatesSharedMemoryNumWarpsElems<<<NBLOCKS, BLOCKSIZE>>>(dev_A, N, dev_globalSum);
  }
  else{
    printf("\nIncorrect Mode: %d. Returning.", MODE);
    return 0;
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
//Mode 4
//Extends mode 3
//Uses atomic updates to shared memory for the sum in the block
__global__ void extendModeThreeWithAtomicUpdatesSharedMemoryOneElems(int * A, const unsigned int N, int * outputSum) {

//Write code here
    __shared__ int sharedSum;

    // Initialize sharedSum to 0 for each block
    if (threadIdx.x == 0) {
        sharedSum = 0;
    }

    __syncthreads();

    // starting position in A based on block ID
    unsigned int offsetIdx = 2 * COARSEFACTOR * blockDim.x * blockIdx.x;

    unsigned int offsetIdxThread = offsetIdx + threadIdx.x;

    int localSum = A[offsetIdxThread];

    for (unsigned int i = 1; i < COARSEFACTOR * 2; i++) {
        localSum += A[offsetIdxThread + i * blockDim.x];
    }

    // Reduction in shared memory
    atomicAdd(&sharedSum, localSum);

    __syncthreads();

    // Write to global memory using threadIdx.x = 0
    if (threadIdx.x == 0) {
        atomicAdd(outputSum, sharedSum);
    }

return;
}

//Mode 5
//Extends mode 3
//Uses atomic updates to shared memory for the sum in the block (uses one element in shared memory per warp in each block)
__global__ void extendModeThreeWithAtomicUpdatesSharedMemoryNumWarpsElems(int * A, const unsigned int N, int * outputSum) {

//Write code here
__shared__ int sharedA[BLOCKSIZE/32];
  int warpLength = 32;  
  //starting position in A based on block ID
  //called "segment" in the textbook
  unsigned int offsetIdx = 2*COARSEFACTOR*blockDim.x*blockIdx.x;

  unsigned int offsetIdxThread = offsetIdx + threadIdx.x;

  int localSum = A[offsetIdxThread];

if(threadIdx.x == 0 ) { 
	for(int i = 0; i < blockDim.x/warpLength ; i++ )
	{
           sharedA[i]=0;
	}
}

  __syncthreads(); 
  for(unsigned int i=1; i<COARSEFACTOR*2; i++){
  localSum += A[offsetIdxThread + i*blockDim.x];
  }
 
    // Reduction in shared memory
    atomicAdd(&sharedA[threadIdx.x / warpLength], localSum);

    __syncthreads();

    if (threadIdx.x == 0) {
       for (int i = 1; i < blockDim.x / warpLength; i++) {
            atomicAdd(&sharedA[0],sharedA[i]);
        }
        atomicAdd(outputSum, sharedA[0]);
    }

return;
}



