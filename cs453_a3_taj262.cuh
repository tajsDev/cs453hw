//If you have device functions, declare 
//device function prototypes here 

 __device__ void pageFIntoRegisters(float * F, float * registersF);


//////////////////////////////////////////////////////////////////////
//Kernels with Global A (4 kernels)
//////////////////////////////////////////////////////////////////////

//Arguments for all kernels except the two constant memory F kernels where F is a global variable 
//and is not accessed through a device pointer

//A- input array
//F- input filter/weights
//B- output array
//r- filter radius for array F (total length of F is 2r+1)
//NUMELEM - NUMELEMS in A and B


//MODE==1
__global__ void convolutionGlobalAGlobalF(float *A, float *F, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
unsigned const int tid = threadIdx.x + (blockIdx.x*blockDim.x); 
  float localTotal = 0 ;
  if(tid < NUMELEM ) {
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = tid+j-r;
      if(idxA >=0 && idxA < NUMELEM ) {
        localTotal+=A[idxA]*F[j];
      }
    }
    B[tid] = localTotal ; 
  }
return;
}

//MODE==2
__global__ void convolutionGlobalASharedF(float *A, float *F, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
 unsigned const int tid = threadIdx.x + (blockIdx.x*blockDim.x); //thread ID
__shared__ float  sharedFilter[RNELEMTOTAL];
  float localTotal = 0 ;

  if(threadIdx.x < RNELEMTOTAL ) { 
      sharedFilter[threadIdx.x] = F[threadIdx.x];
    }

  __syncthreads();

  
  if(tid < NUMELEM ) {
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = tid+j-r;
      if(idxA >=0 && idxA < NUMELEM ) {
        localTotal+=A[idxA]*sharedFilter[j];
      }
    }
    B[tid] = localTotal ; 
  }


return;
}


//MODE==3
__global__ void convolutionGlobalARegistersF(float *A, float *F, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
 unsigned const int tid = threadIdx.x + (blockIdx.x*blockDim.x); //thread ID
  if( tid < NUMELEM ) {
    float localTotal = 0 ;
    float localF[RNELEMTOTAL] ;
    pageFIntoRegisters(F,localF);
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = tid+j-r;
      if(idxA >=0 && idxA < NUMELEM ) {
        localTotal+=A[idxA]*localF[j];
      }
    }
    B[tid] = localTotal ; 

  }


return;
}


//MODE==4
__global__ void convolutionGlobalAConstantF(float *A, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
unsigned const int tid = threadIdx.x + (blockIdx.x*blockDim.x); //thread ID

//Write code here
  if( tid < NUMELEM ) {
    float localTotal = 0 ;
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = tid+j-r;
      if(idxA >=0 && idxA < NUMELEM ) {
        localTotal+=A[idxA]*FGLOBAL[j];
      }
    }
    B[tid] = localTotal ; 

  }

return;
}



//////////////////////////////////////////////////////////////////////
//Kernels with shared A (4 kernels)
//////////////////////////////////////////////////////////////////////

//Four kernels where each block pages the required range of A into shared memory

//The blocks assigned output elements that access the bounds of the array default to using global memory
//the blocks only requiring "internal" elements in A use shared memory
//(This makes programming the assignment easier than dealing with all of the boundary conditions.)

//MODE==5
__global__ void convolutionSharedAGlobalF(float *A, float *F, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
  int tid = threadIdx.x + (blockIdx.x*blockDim.x); //thread ID
__shared__ float sharedA[TILESIZE + 2*R];
//Write code here

    if (tid < NUMELEM) { // Load data into shared memory
       sharedA[threadIdx.x] = A[tid];
	if(threadIdx.x < 2 * R ) {
		sharedA[threadIdx.x + blockDim.x] = A[tid + blockDim.x];
	} 
        
    }
  __syncthreads();

  float localTotal = 0 ;
if(MODE == 5 ) {
  if(tid < NUMELEM ) {
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = threadIdx.x+j;
        localTotal = localTotal + sharedA[idxA]*F[j];
    }
    B[tid] = localTotal ; 
  }
}
else {
 if(tid < NUMELEM ) {
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = threadIdx.x+j;
        localTotal = localTotal + A[idxA]*F[j];
    }
    B[tid] = localTotal ; 
  }


}
return;
}


//MODE==6
__global__ void convolutionSharedASharedF(float *A, float *F, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
unsigned const int tid = threadIdx.x + (blockIdx.x*blockDim.x); //thread ID
__shared__ float sharedA[TILESIZE + 2*R];
//Write code here


__shared__ float sharedFilter[RNELEMTOTAL];
  float localTotal = 0 ;

    if (tid < NUMELEM) { // Load data into shared memory
       sharedA[threadIdx.x] = A[tid];
	if(threadIdx.x < 2 * R ) {
		sharedA[threadIdx.x + blockDim.x] = A[tid + blockDim.x];
	} 
        
    }

  __syncthreads();

  if(threadIdx.x < RNELEMTOTAL ) { 
      sharedFilter[threadIdx.x] = F[threadIdx.x];
    }

   __syncthreads();

  
  if(tid < NUMELEM ) {
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = threadIdx.x+j;
        localTotal+=sharedA[idxA]*sharedFilter[j];
    }
    B[tid] = localTotal ; 
  }


return;
}

//MODE==7
__global__ void convolutionSharedARegistersF(float *A, float *F, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	
 unsigned const int tid = threadIdx.x + (blockIdx.x*blockDim.x); //thread ID
__shared__ float sharedA[TILESIZE + 2*R];
//Write code here
float localF[RNELEMTOTAL];
float localTotal = 0 ;
    if (tid < NUMELEM) { // Load data into shared memory
       sharedA[threadIdx.x] = A[tid];
	if(threadIdx.x < 2 * R ) {
		sharedA[threadIdx.x + blockDim.x] = A[tid + blockDim.x];
	} 
        
    }


  __syncthreads();

  if( tid < NUMELEM ) {
    pageFIntoRegisters(F,localF);
    for(unsigned int j=0;j<2*r+1; j ++ ) {
        unsigned int idxA = threadIdx.x+j;
        localTotal+=sharedA[idxA]*localF[j];
    }
    B[tid] = localTotal ; 

  }

return;
}


//MODE==8
__global__ void convolutionSharedAConstantF(float *A, float *B, const unsigned int r,  const unsigned int NUMELEM) {

//thread ID corresponding to output element in B
//Be sure to uncomment (currently commented to avoid compiler warnings)	

  unsigned const int tid = threadIdx.x + (blockIdx.x * blockDim.x); // Thread ID
    __shared__ float sharedA[TILESIZE + 2 * R];
	float localTotal = 0 ;
    if (tid < NUMELEM) { // Load data into shared memory
       sharedA[threadIdx.x] = A[tid];
	if(threadIdx.x < 2 * R ) {
		sharedA[threadIdx.x + blockDim.x] = A[tid + blockDim.x];
	} 
        
    }

    __syncthreads();

 if( tid < NUMELEM ) {
     for(unsigned int j=0;j<2*r+1; j ++ ) {
         unsigned int idxA = threadIdx.x+j;
       
         localTotal+=sharedA[idxA]*FGLOBAL[j];
       
     }
     B[tid] = localTotal ;
 
   }

return;
}



///////////////////////////////////
//Device functions (optional)
///////////////////////////////////

//You may decide to use functions so that you do not duplicate too much code. 
//Below is an example function that I used in my solution (you are not required to use it).

//copy F into thread's local registers
 __device__ void pageFIntoRegisters(float * F, float * registersF) {
    for (unsigned int i=0; i<RNELEMTOTAL; i++) {
	  	registersF[i]=F[i];
 	}	
  return ; 
 }
