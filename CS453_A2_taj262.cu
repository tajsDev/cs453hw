//Matrix multiplication that uses
//1-D A, B, C on both GPU and CPU (use FP32)


//CPU- reference implementation

//GPU- one output element per thread (MODE==1) //this is the original in matrix_multiply.cu
//GPU- one output element per thread with shared-memory tiling (MODE==4) //this is the original in matrix_multiply.cu

//Transposed matrix B:
//Same as MODE==1 above but matrix B is transposed (MODE==5)
//Same as MODE==4 above but matrix B is transposed (MODE==6)

//GPU- one element per thread but transpose matrix B so that we have more coalesced memory accesses (MODE==6)


#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>

#include <math.h>


//You may want to move these to your job script
#define N 2048 //max 2048 (for demonstrations in class), use powers of 2 for division 
				//into shared memory tiles for MODE==4

#define MODE 5 //see implementations above for associated modes

#define BLOCKDIMTILE 32     


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
void warmUpGPU();
void compareMatrices(float * C_GPU, float * C_CPU, unsigned int NUMELEM);
void printMatrixIfSmall(float * C, const unsigned int NUMELEM, bool CPUGPU);
void computeMatrixCPU(float * A, float * B, float *C, const unsigned int NUMELEM);
void outputSumElems(float * C, unsigned int NUMELEM);

__global__ void matrixMultiOneElemPerThread(float *A, float *B, float *C, const unsigned int NUMELEM);
__global__ void matrixMultiOneElemPerThreadTransposedMatrixB(float *A, float *B, float *C, const unsigned int NUMELEM);
__global__ void matrixMultiOneElemPerThreadSharedMemoryTile(float *A, float *B, float *C, const unsigned int NUMELEM);
__global__ void matrixMultiOneElemPerThreadSharedMemoryTileTransposedB(float *A, float *B, float *C, const unsigned int NUMELEM);

__global__ void matrixMultiOneRowPerThread(float *A,float *B,float *C,const unsigned int NUMELEM);

__global__ void matrixMultiOneColumnPerThread(float *A,float *B,float *C, const unsigned int NUMELEM); 

int main(int argc, char *argv[])
{

	
	warmUpGPU();
	
	//change OpenMP settings
	//disregard --- only for parallel CPU version if applicable
	omp_set_num_threads(1);

  	//seed random number generator with constant seed
  	srand(123);  

  	float * A;
  	float * B;
  	float * B_Transposed; 
  	float * C;
  	float * C_CPU;

  	A=(float *)malloc(sizeof(float)*N*N);
  	B=(float *)malloc(sizeof(float)*N*N);
  	B_Transposed=(float *)malloc(sizeof(float)*N*N);
  	C=(float *)calloc(N*N,sizeof(float));
  	C_CPU=(float *)calloc(N*N,sizeof(float));


  	//init matrices of FP32 with random numbers between 0 and 1
  	for (unsigned int i=0; i<N*N; i++){
  		A[i]=(float)rand()/(float)RAND_MAX;
  		B[i]=(float)rand()/(float)RAND_MAX;

  		//Transposed matrix
  		//Write code here to copy B into B_Transposed to populate the transposed matrix
  	}
	for(unsigned int j=0; j < N ; j ++ ) {
		for(unsigned int k= 0; k < N ; k++ ) {
			B_Transposed[j + k * N] = B[j *N +k];
		}
	}
	printf("\nMemory requested for 5x NxN matrices (GiB) %f", (5.0*N*N*sizeof(float)/(1024.0*1024.0*1024.0)));



	///////////////////////////
	//CPU version:
	///////////////////////////

	printf("\nCommented sequential CPU execution");
	//computeMatrixCPU(A, B, C_CPU, N);
	
	//print matrix if N is <= 10x10
	//	printMatrixIfSmall(C_CPU, N, 0);
	
	
	/////////////////////////////
	//GPU
	////////////////////////////	

	
	double tstart=omp_get_wtime();

	
	cudaError_t errCode=cudaSuccess;
	
	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl; 	
	}

	float * dev_A;
	float * dev_B;
	float * dev_C;
	
	unsigned int * debug;
	debug=(unsigned int *)malloc(sizeof(unsigned int));
	*debug=0;

	//allocate on the device: A, B, C
	gpuErrchk(cudaMalloc((float**)&dev_A, sizeof(float)*N*N));	
	gpuErrchk(cudaMalloc((float**)&dev_B, sizeof(float)*N*N));	
	gpuErrchk(cudaMalloc((float**)&dev_C, sizeof(float)*N*N));	
	
	//copy A to device
	gpuErrchk(cudaMemcpy(dev_A, A, sizeof(float)*N*N, cudaMemcpyHostToDevice));
	
	if (MODE==1 || MODE==4 || MODE == 2 || MODE == 3 )
	{
		gpuErrchk(cudaMemcpy(dev_B, B, sizeof(float)*N*N, cudaMemcpyHostToDevice));
	}

	//copy B to device (transposed)
	else if (MODE==5 || MODE==6)
	{
		gpuErrchk(cudaMemcpy(dev_B, B_Transposed, sizeof(float)*N*N, cudaMemcpyHostToDevice));
	}

	//copy C to device (initialized to 0)
	gpuErrchk(cudaMemcpy(dev_C, C, sizeof(float)*N*N, cudaMemcpyHostToDevice));


	//execute kernel

	//MODE==1 refers to one thread per output element of the matrix
	if (MODE==1){
		printf("\nMODE==1");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		unsigned int BLOCKDIM = 32; //blocks are of size 32x32=1024
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);
		
		matrixMultiOneElemPerThread<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	}
	else if(MODE == 2 ) {

		printf("\nMODE==2");
		//set number of blocks 
		//unlike MODE==1, we only use 1-D grid because we are assigning one thread per row
		unsigned int BLOCKDIM = 128; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		matrixMultiOneRowPerThread<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	//execute kernel
	}
	else if(MODE == 3 ) { 
		printf("\nMODE==3");
		//set number of blocks 
		//unlike MODE==1, we only use 1-D grid because we are assigning one thread per row
		unsigned int BLOCKDIM = 128; 
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);
		
		matrixMultiOneColumnPerThread<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);


	}
	else if (MODE==4){
		printf("\nMODE==4");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		//also for convenience, set the number of threads per block to be the shared memory tile size
		unsigned int BLOCKDIM = BLOCKDIMTILE; //blocks are of size BLOCKDIMTILE*BLOCKDIMTILE
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);
		
		matrixMultiOneElemPerThreadSharedMemoryTile<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	}
	//MODE==5 refers to one thread per output element of the matrix
	//uses transposed matrix B for coalesced global memory accesses
	else if (MODE==5){
		printf("\nMODE==5");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		unsigned int BLOCKDIM = 32; //blocks are of size 32x32=1024
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);
		
		matrixMultiOneElemPerThreadTransposedMatrixB<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	}
	//MODE==6 refers to one thread per output element of the matrix
	//where shared-memory tiling is used, but works on the transposed matrix
	//similar to MODE==4
	//1-D grid (for the rows)
	else if (MODE==6){
		printf("\nMODE==6");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		//also for convenience, set the number of threads per block to be the shared memory tile size
		unsigned int BLOCKDIM = BLOCKDIMTILE; //blocks are of size BLOCKDIMTILE*BLOCKDIMTILE
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);
		
		matrixMultiOneElemPerThreadSharedMemoryTileTransposedB<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	}
	else
	{
		printf("Error: incorrect mode\n");
		return 0;
	}


	//check kernel errors
	errCode=cudaGetLastError();
	if(errCode != cudaSuccess) {
	cout << "\nError: GPU kernel had an error with code: " << errCode << endl; 
	}

	//end execute kernel

	//Copy C from the GPU
	gpuErrchk(cudaMemcpy(C, dev_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost));
	
	double tend=omp_get_wtime();
	
	printf("\nTotal time GPU (s): %f",tend-tstart);
	
	printf("\nThis is for the sum of the CPU");
	//outputSumElems(C_CPU,N*N);
	//print sum of elements in GPU array C
	outputSumElems(C, N*N);

	//print matrix if N is less than 16
	printMatrixIfSmall(C, N, 1);
	
	//Compare CPU and GPU matrices to determine if there are errors in floating point arithmetic or in the GPU code
	compareMatrices(C, C_CPU, N*N);


	printf("\n");

	//free memory
	free(A);
  	free(B);
  	free(B_Transposed);
  	free(C);
  	free(C_CPU);
  	cudaFree(dev_A);
  	cudaFree(dev_B);
  	cudaFree(dev_C);

	return 0;
}


void outputSumElems(float * C, unsigned int NUMELEM)
{
	float sumElems=0;
	for (int i=0; i<NUMELEM; i++)
	{
		sumElems += C[i];
	}
	printf("\nSum of elems in GPU output matrix (C): %f",sumElems);
}

void compareMatrices(float * C_GPU, float * C_CPU, unsigned int NUMELEM)
{
	float sumDelta=0;
	float maxDelta=0; //keep track of the maximum difference
	for (int i=0; i<NUMELEM; i++)
	{
		float delta = fabs(C_CPU[i]-C_GPU[i]);
		sumDelta += delta;
		if(maxDelta<delta)
		{
			maxDelta = delta;
		}
	}

	printf("\nSum of deltas between matrices: %f",sumDelta);
	printf("\nMaximum delta between elements between matrices: %f",maxDelta);
}




void warmUpGPU(){
printf("\nWarming up GPU for time trialing...\n");	
cudaDeviceSynchronize();
return;
}


void computeMatrixCPU(float * A, float * B, float * C, const unsigned int NUMELEM)
{
	double tstartcpu=omp_get_wtime();

	int ROW=0;
	int COL=0;

	for (ROW=0; ROW<N; ROW++)
		for (COL=0; COL<N; COL++)
			for (int k=0; k<N; k++)
			{
				C[(ROW*N)+COL]+=A[ROW*N+k]*B[COL+(k*N)];
			}

	double tendcpu=omp_get_wtime();
	printf("\nTime CPU: %f",tendcpu - tstartcpu);
}

void printMatrixIfSmall(float * C, const unsigned int NUMELEM, bool CPUGPU)
{
	int i, j;
	int cnt=0;
	if (N<=16)
	{
		if(CPUGPU==0)
			printf("\nCPU matrix is: \n");
		else
			printf("\nGPU matrix is: \n");
		
		for (i=0; i<NUMELEM; i++){
			for (j=0; j<NUMELEM; j++){
				printf("%.2f, ",C[cnt]);
				cnt++;
			}
			printf("\n");
		}
	}
}


//matrix multiply
//each thread computes a single element of C using a row of A and column of B
__global__ void matrixMultiOneElemPerThread(float *A, float *B, float *C, const unsigned int NUMELEM) {

//copy code from in-class activity
const unsigned int row = threadIdx.x + ( blockIdx.x * blockDim.x) ;
const unsigned int col = threadIdx.y + ( blockIdx.y * blockDim.y);
if ( row < NUMELEM && col < NUMELEM ) {
for(int k = 0 ; k < NUMELEM ; k ++ ) {
	C[row * NUMELEM  + col ]+= A[row * NUMELEM + k] * B[k* NUMELEM +col];
}
}
return;
}
__global__ void matrixMultiOneRowPerThread(float *A, float *B , float *C, const unsigned int NUMELEM ) {
	const unsigned int row = threadIdx.x + ( blockIdx.x * blockDim.y );
	if( row < NUMELEM ) {
	for(int col  =0 ; col  < NUMELEM ; col++ ) {

		for(int k = 0; k < NUMELEM  ; k++ ) { 
			C[row*NUMELEM + col]+= A[row * NUMELEM + k] * B[k * NUMELEM + col];
		}
	}

	}

}

__global__ void matrixMultiOneColumnPerThread(float *A, float *B,float *C,const unsigned int NUMELEM ) {

	const unsigned int col = threadIdx.x + ( blockIdx.x * blockDim.x );

	if(col < NUMELEM ) { 
	for(int row = 0 ; row < NUMELEM ; row ++ ) { 
		for(int k = 0 ; k < NUMELEM ; k++ ) {
			C[row*NUMELEM+col]+= A[row * NUMELEM + k] * B[k * NUMELEM + col];
		}
	}

}
}
//matrix multiply
//each thread computes a single element of C using a row of A and column of B
//uses shared memory to tile the computation to eliminate extra accesses to global memory
//This example is from Chapter 5 in the textbook with some minor modifications
__global__ void matrixMultiOneElemPerThreadSharedMemoryTile(float *A, float *B, float *C, const unsigned int NUMELEM) {

//Copy code from in-class activity
const unsigned int row  = threadIdx.x + ( blockIdx.x * blockDim.x ); 
const unsigned int col = threadIdx.y + ( blockIdx.y * blockDim.y ) ;

float sum = 0 ;
__shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
__shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];
//tile iteration starts at zero and iterate through tile
for(int i = 0 ; i < NUMELEM ; i+=BLOCKDIMTILE ) {
	//grad the values of x and y for tileA and tileB(rows and cols)
	tileA[threadIdx.x][threadIdx.y] = A[row*NUMELEM+i+threadIdx.y]; 
	tileB[threadIdx.x][threadIdx.y] = B[col+(NUMELEM*(i+threadIdx.x))];
	
	__syncthreads();
	//iterate over the tile
	for(int k = 0 ; k < BLOCKDIMTILE ; k++ ) {
		//add to sum from tile A and tile B value
		sum  += tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
	}

	__syncthreads();
}
	//return local memry to global memry
	C[row*NUMELEM +col] = sum;
return;
}

//matrix multiply
//each thread computes a single element of C using a row of A and a row of B
//Matrix B is transposed to allow coalesced memory accesses
__global__ void matrixMultiOneElemPerThreadTransposedMatrixB(float *A, float *B, float *C, const unsigned int NUMELEM) {

//write code here (copy/paste from MODE==1 to get started)
const unsigned int row  = threadIdx.x + ( blockIdx.x * blockDim.x);
const unsigned int col = threadIdx.y + ( blockIdx.y * blockDim.y);
if(row < NUMELEM && col < NUMELEM  ) {
	for(int k = 0 ; k < NUMELEM ; k++ ) {
 	C[row * NUMELEM  + col ]+= A[row * NUMELEM + k] * B[k + col * NUMELEM ];
	}
} 
return;
}

//matrix multiply
//each thread computes a single element of C using a row of A and column of B
//uses shared memory to tile the computation to eliminate extra accesses to global memory
//This example is from Chapter 5 in the textbook with some minor modifications
__global__ void matrixMultiOneElemPerThreadSharedMemoryTileTransposedB(float *A, float *B, float *C, const unsigned int NUMELEM) {

//write code here (copy/paste from MODE==4 to get started)
const unsigned int row = threadIdx.x + (blockIdx.x * blockDim.x);
const unsigned int col = threadIdx.y + (blockIdx.y * blockDim.y);
float sum = 0 ;

__shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
__shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

for(int phase = 0; phase < NUMELEM ; phase+=BLOCKDIMTILE ) {
	tileA[threadIdx.x][threadIdx.y] = A[row*NUMELEM+phase+threadIdx.y];
	tileB[threadIdx.x][threadIdx.y] = B[col*NUMELEM+phase+threadIdx.x];

	__syncthreads();
	for(int k = 0 ; k < BLOCKDIMTILE; k++ ) {

		sum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
	}


	__syncthreads();
} 
	C[row*NUMELEM+col] = sum; 
return;
}







