//You may prefer to comment N, R, RNELEMTOTAL, and MODE and move them to your job script.
//When you submit your assignment, ensure that you use these values
//and not other values that you may have used when implementing/debugging
//your program. Also, ensure that these are uncommented when you submit.

//#define N 200000000 //length of the input and output arrays
//#define R 5 //filter array length (total length is 2*R+1)
//#define RNELEMTOTAL 11 //total length of filter
//#define MODE 1  //see implementations above for associated modes

//1-D block size for all kernels
#define BLOCKSIZE 512

//for implementations that use shared memory for input array A (Modes 5-8)
#define TILESIZE 512

//For modes 4 and 8
__constant__ float FGLOBAL[RNELEMTOTAL];

